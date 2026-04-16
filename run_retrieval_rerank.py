#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


WORKSPACE_ROOT = Path("/workspace")
QWEN_ROOT = WORKSPACE_ROOT / "Qwen3-VL-Embedding"
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage retrieval pipeline: Qwen3-VL-Embedding recall + Qwen3-VL-Reranker rerank."
    )
    parser.add_argument("--video-root", type=Path, default=WORKSPACE_ROOT / "test")
    parser.add_argument("--query-json", type=Path, default=WORKSPACE_ROOT / "ucf_crime_ar_test.json")
    parser.add_argument("--description-json", type=Path, default=WORKSPACE_ROOT / "video_description.json")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "retrieval_rerank" / "outputs")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--reranker-model", type=str, default="Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument("--top-k-retrieval", type=int, default=50)
    parser.add_argument("--final-mode", choices=["rerank", "fusion"], default="rerank")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--embedding-batch-size", type=int, default=6)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_videos(video_root: Path) -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [
        p
        for p in video_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and "__MACOSX" not in p.parts
        and not p.name.startswith(".")
        and not p.name.startswith("._")
    ]
    videos.sort()
    return videos


def build_description_map(raw_descriptions: Any) -> Dict[str, str]:
    if isinstance(raw_descriptions, dict):
        return {str(k): str(v) for k, v in raw_descriptions.items()}

    description_map: Dict[str, str] = {}
    if isinstance(raw_descriptions, list):
        for item in raw_descriptions:
            if not isinstance(item, dict):
                continue
            name = item.get("video_name") or item.get("Video Name") or item.get("name")
            desc = item.get("video_description") or item.get("description") or item.get("text")
            if name and desc:
                description_map[Path(str(name)).stem] = str(desc)
    return description_map


def normalize_queries(raw_queries: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_queries, dict):
        raw_queries = raw_queries.get("queries", [raw_queries])

    queries: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            continue
        text = item.get("English Text") or item.get("query") or item.get("text")
        if not text:
            continue
        gt_video = item.get("Video Name") or item.get("video_name") or item.get("video")
        queries.append(
            {
                "query_id": item.get("query_id", idx),
                "query_text": str(text),
                "ground_truth_video": str(gt_video) if gt_video else None,
            }
        )
    return queries


def build_video_records(video_paths: List[Path], description_map: Dict[str, str], video_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, video_path in enumerate(video_paths):
        records.append(
            {
                "video_id": idx,
                "video_path": str(video_path),
                "video_name": video_path.name,
                "video_stem": video_path.stem,
                "relative_path": str(video_path.relative_to(video_root)),
                "description": description_map.get(video_path.stem),
            }
        )
    return records


def get_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def load_or_compute_corpus_embeddings(
    embedder: Qwen3VLEmbedder,
    video_records: List[Dict[str, Any]],
    cache_path: Path,
    batch_size: int,
    fps: float,
    max_frames: int,
    overwrite_cache: bool,
) -> torch.Tensor:
    if cache_path.exists() and not overwrite_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("video_paths", []) == [record["video_path"] for record in video_records]:
            print(f"[Cache] Loaded corpus embeddings from {cache_path}")
            return payload["embeddings"].float()
        print("[Cache] Video list changed, recomputing corpus embeddings.")

    all_embeddings: List[torch.Tensor] = []
    total_batches = math.ceil(len(video_records) / batch_size)
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(video_records))
        batch = video_records[start:end]
        print(f"[Embedding] Encoding videos {start + 1}-{end}/{len(video_records)}")
        embeddings = embedder.process(
            [
                {
                    "video": record["video_path"],
                    "instruction": "Retrieve videos relevant to the user's query.",
                    "fps": fps,
                    "max_frames": max_frames,
                }
                for record in batch
            ]
        )
        all_embeddings.append(embeddings.detach().cpu().float())

    corpus_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(
        {
            "video_paths": [record["video_path"] for record in video_records],
            "embeddings": corpus_embeddings,
        },
        cache_path,
    )
    print(f"[Cache] Saved corpus embeddings to {cache_path}")
    return corpus_embeddings


def encode_query(embedder: Qwen3VLEmbedder, query_text: str) -> torch.Tensor:
    embedding = embedder.process(
        [
            {
                "text": query_text,
                "instruction": "Retrieve videos relevant to the user's query.",
            }
        ]
    )
    return embedding.detach().cpu().float()[0]


def build_rerank_documents(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for candidate in candidates:
        doc: Dict[str, Any] = {"video": candidate["video_path"]}
        if candidate.get("description"):
            doc["text"] = candidate["description"]
        documents.append(doc)
    return documents


def compute_final_score(visual_score: float, rerank_score: float, final_mode: str, alpha: float) -> float:
    if final_mode == "fusion":
        return alpha * visual_score + (1.0 - alpha) * rerank_score
    return rerank_score


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ground_truth_matches(candidate_relative_path: str, ground_truth_video: Optional[str]) -> bool:
    if not ground_truth_video:
        return False

    gt = str(Path(ground_truth_video))
    candidate = str(Path(candidate_relative_path))
    if candidate == gt:
        return True

    # Query annotations for anomaly videos may omit the leading
    # "Testing_Anomaly_Videos/" prefix used by the indexed corpus.
    return candidate.endswith("/" + gt)


def process_queries(
    queries: List[Dict[str, Any]],
    video_records: List[Dict[str, Any]],
    corpus_embeddings: torch.Tensor,
    embedder: Qwen3VLEmbedder,
    reranker: Qwen3VLReranker,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stage1_results: List[Dict[str, Any]] = []
    stage2_results: List[Dict[str, Any]] = []
    final_results: List[Dict[str, Any]] = []

    for query_idx, query in enumerate(queries, start=1):
        query_text = query["query_text"]
        print(f"[Query {query_idx}/{len(queries)}] {query_text}")

        query_embedding = encode_query(embedder, query_text)
        scores = torch.matmul(corpus_embeddings, query_embedding)
        top_scores, top_indices = torch.topk(scores, k=min(args.top_k_retrieval, scores.shape[0]))

        candidates: List[Dict[str, Any]] = []
        for rank_idx, (video_idx, visual_score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), start=1):
            candidate = dict(video_records[video_idx])
            candidate["visual_rank"] = rank_idx
            candidate["visual_score"] = float(visual_score)
            candidates.append(candidate)

        stage1_results.append(
            {
                "query_id": query["query_id"],
                "query_text": query_text,
                "ground_truth_video": query.get("ground_truth_video"),
                "top_k_retrieval": args.top_k_retrieval,
                "candidates": [
                    {
                        "video_path": candidate["video_path"],
                        "relative_path": candidate["relative_path"],
                        "video_name": candidate["video_name"],
                        "video_stem": candidate["video_stem"],
                        "description": candidate.get("description"),
                        "visual_rank": candidate["visual_rank"],
                        "visual_score": candidate["visual_score"],
                    }
                    for candidate in candidates
                ],
            }
        )

        gt_video = query.get("ground_truth_video")
        gt_stage1_rank = None
        gt_stage1_hit = False
        if gt_video:
            for candidate in candidates:
                if ground_truth_matches(candidate["relative_path"], gt_video):
                    gt_stage1_rank = candidate["visual_rank"]
                    gt_stage1_hit = True
                    break
        stage1_results[-1]["ground_truth_in_top_k"] = gt_stage1_hit
        stage1_results[-1]["ground_truth_stage1_rank"] = gt_stage1_rank

        rerank_scores = reranker.process(
            {
                "instruction": "Retrieve relevant videos for the user's query.",
                "query": {"text": query_text},
                "documents": build_rerank_documents(candidates),
                "fps": args.fps,
                "max_frames": args.max_frames,
            }
        )

        stage2_candidates: List[Dict[str, Any]] = []
        final_candidates: List[Dict[str, Any]] = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            stage2_candidate = {
                "video_path": candidate["video_path"],
                "relative_path": candidate["relative_path"],
                "video_name": candidate["video_name"],
                "video_stem": candidate["video_stem"],
                "description": candidate.get("description"),
                "visual_rank": candidate["visual_rank"],
                "visual_score": candidate["visual_score"],
                "rerank_score": float(rerank_score),
            }
            stage2_candidates.append(stage2_candidate)
            final_candidates.append(
                {
                    "video_path": candidate["video_path"],
                    "relative_path": candidate["relative_path"],
                    "video_name": candidate["video_name"],
                    "video_stem": candidate["video_stem"],
                    "description": candidate.get("description"),
                    "visual_rank": candidate["visual_rank"],
                    "visual_score": candidate["visual_score"],
                    "rerank_score": float(rerank_score),
                    "final_score": float(
                        compute_final_score(candidate["visual_score"], float(rerank_score), args.final_mode, args.alpha)
                    ),
                }
            )

        stage2_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        for rerank_rank, candidate in enumerate(stage2_candidates, start=1):
            candidate["rerank_rank"] = rerank_rank

        final_candidates.sort(key=lambda item: item["final_score"], reverse=True)
        for final_rank, candidate in enumerate(final_candidates, start=1):
            candidate["final_rank"] = final_rank

        gt_final_rank = None
        gt_in_top_k = False
        if gt_video:
            for candidate in final_candidates:
                if ground_truth_matches(candidate["relative_path"], gt_video):
                    gt_final_rank = candidate["final_rank"]
                    gt_in_top_k = True
                    break

        stage2_results.append(
            {
                "query_id": query["query_id"],
                "query_text": query_text,
                "ground_truth_video": gt_video,
                "top_k_retrieval": args.top_k_retrieval,
                "candidates": stage2_candidates,
            }
        )

        final_results.append(
            {
                "query_id": query["query_id"],
                "query_text": query_text,
                "ground_truth_video": gt_video,
                "ground_truth_in_top_k": gt_in_top_k,
                "ground_truth_final_rank": gt_final_rank,
                "top_k_retrieval": args.top_k_retrieval,
                "final_mode": args.final_mode,
                "alpha": args.alpha,
                "candidates": final_candidates,
            }
        )

    hit_count = sum(1 for item in final_results if item["ground_truth_in_top_k"])
    return {
        "summary": {
            "num_queries": len(final_results),
            "num_videos": len(video_records),
            "top_k_retrieval": args.top_k_retrieval,
            "final_mode": args.final_mode,
            "alpha": args.alpha,
            "retrieval_hit_rate_at_k": (hit_count / len(final_results)) if final_results else 0.0,
        },
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "final_results": final_results,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    queries = normalize_queries(load_json(args.query_json))
    if not queries:
        raise ValueError(f"No valid queries found in {args.query_json}")

    description_map: Dict[str, str] = {}
    if args.description_json.exists():
        description_map = build_description_map(load_json(args.description_json))

    video_paths = discover_videos(args.video_root)
    if not video_paths:
        raise ValueError(f"No videos found under {args.video_root}")

    video_records = build_video_records(video_paths, description_map, args.video_root)
    model_kwargs = get_model_kwargs(args)

    print("[Init] Loading embedding model...")
    embedder = Qwen3VLEmbedder(args.embedding_model, **model_kwargs)
    print("[Init] Loading reranker model...")
    reranker = Qwen3VLReranker(args.reranker_model, **model_kwargs)

    corpus_embeddings = load_or_compute_corpus_embeddings(
        embedder=embedder,
        video_records=video_records,
        cache_path=args.output_dir / "video_embeddings.pt",
        batch_size=args.embedding_batch_size,
        fps=args.fps,
        max_frames=args.max_frames,
        overwrite_cache=args.overwrite_cache,
    )

    payload = process_queries(queries, video_records, corpus_embeddings, embedder, reranker, args)

    save_json(
        args.output_dir / "run_config.json",
        {
            "video_root": str(args.video_root),
            "query_json": str(args.query_json),
            "description_json": str(args.description_json),
            "output_dir": str(args.output_dir),
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,``
            "top_k_retrieval": args.top_k_retrieval,
            "final_mode": args.final_mode,
            "alpha": args.alpha,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "embedding_batch_size": args.embedding_batch_size,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
    )
    save_json(args.output_dir / "video_index.json", video_records)
    save_json(args.output_dir / "summary.json", payload["summary"])
    save_json(args.output_dir / "stage1_retrieval.json", payload["stage1_results"])
    save_json(args.output_dir / "stage2_rerank.json", payload["stage2_results"])
    save_json(args.output_dir / "final_results.json", payload["final_results"])
    save_json(args.output_dir / "retrieval_rerank_results.json", payload)

    print(f"[Done] Results saved to {args.output_dir}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
