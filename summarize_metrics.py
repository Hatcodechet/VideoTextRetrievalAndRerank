#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize retrieval/rerank outputs into IR-style metrics."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("/workspace/retrieval_rerank/outputs/final_results.json"),
        help="Path to final_results.json or stage1_retrieval.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the concise metrics JSON. Defaults beside results file.",
    )
    parser.add_argument(
        "--detailed-output-json",
        type=Path,
        default=None,
        help="Optional path to save the detailed metrics JSON with per-category results.",
    )
    parser.add_argument(
        "--rank-key",
        type=str,
        default=None,
        help="Optional explicit rank key, e.g. final_rank or visual_rank.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    return candidate.endswith("/" + gt)


def infer_rank_key(results: List[Dict[str, Any]], explicit_rank_key: Optional[str]) -> str:
    if explicit_rank_key:
        return explicit_rank_key
    if not results or not results[0].get("candidates"):
        raise ValueError("Results file is empty or missing candidates.")

    sample = results[0]["candidates"][0]
    for key in ("final_rank", "rerank_rank", "visual_rank"):
        if key in sample:
            return key
    raise ValueError("Could not infer rank key from candidates.")


def hit_at(ranks: List[int], k: int, total: int) -> float:
    if total == 0:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / total


def summarize_bucket(items: List[Dict[str, Any]], rank_key: str) -> Dict[str, Any]:
    total = len(items)
    matched_ranks: List[int] = []
    missing = 0

    for item in items:
        gt = item.get("ground_truth_video")
        rank = None
        for candidate in item.get("candidates", []):
            if ground_truth_matches(candidate["relative_path"], gt):
                rank = int(candidate[rank_key])
                break
        if rank is None:
            missing += 1
        else:
            matched_ranks.append(rank)

    summary: Dict[str, Any] = {
        "num_queries": total,
        "num_matches": len(matched_ranks),
        "num_misses": missing,
        "hit_at_1": hit_at(matched_ranks, 1, total),
        "hit_at_5": hit_at(matched_ranks, 5, total),
        "hit_at_10": hit_at(matched_ranks, 10, total),
        "hit_at_50": hit_at(matched_ranks, 50, total),
    }
    if matched_ranks:
        summary["mean_rank"] = statistics.mean(matched_ranks)
        summary["median_rank"] = statistics.median(matched_ranks)
    else:
        summary["mean_rank"] = None
        summary["median_rank"] = None
    return summary


def category_name(ground_truth_video: Optional[str]) -> str:
    if not ground_truth_video:
        return "UNKNOWN"
    parts = Path(ground_truth_video).parts
    if not parts:
        return "UNKNOWN"
    return parts[0]


def main() -> None:
    args = parse_args()
    results = load_json(args.results_json)
    if not isinstance(results, list):
        raise ValueError(f"Expected a list of query results in {args.results_json}")

    rank_key = infer_rank_key(results, args.rank_key)
    overall = summarize_bucket(results, rank_key)

    per_category: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(category_name(item.get("ground_truth_video")), []).append(item)
    for category, items in sorted(grouped.items()):
        per_category[category] = summarize_bucket(items, rank_key)

    detailed_payload = {
        "results_json": str(args.results_json),
        "rank_key": rank_key,
        "overall": overall,
        "per_category": per_category,
    }

    output_json = args.output_json or args.results_json.with_name(
        args.results_json.stem + "_metrics.json"
    )
    concise_payload = {
        "results_json": str(args.results_json),
        "rank_key": rank_key,
        **overall,
    }
    save_json(output_json, concise_payload)

    detailed_output_json = args.detailed_output_json or args.results_json.with_name(
        args.results_json.stem + "_metrics_detailed.json"
    )
    save_json(detailed_output_json, detailed_payload)

    print(json.dumps(concise_payload, ensure_ascii=False, indent=2))
    print(f"[Done] Metrics saved to {output_json}")
    print(f"[Done] Detailed metrics saved to {detailed_output_json}")


if __name__ == "__main__":
    main()
