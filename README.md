# Retrieval + VL Rerank

Two-stage video retrieval pipeline for `/workspace/test`:

1. Stage 1 keeps `Qwen3-VL-Embedding-2B` for visual retrieval.
2. Stage 2 adds `Qwen3-VL-Reranker-2B` to rerank the top-50 candidates.
3. Final ranking can use reranker score directly or visual/rerank fusion.

## Inputs

- Videos: `/workspace/test`
- Queries: `/workspace/ucf_crime_ar_test.json`
- Video descriptions: `/workspace/video_description.json`

## Output

Default output directory:

- `/workspace/retrieval_rerank/outputs`

Generated files:

- `video_embeddings.pt`
- `run_config.json`
- `video_index.json`
- `summary.json`
- `stage1_retrieval.json`
- `stage2_rerank.json`
- `final_results.json`
- `retrieval_rerank_results.json`

## Run

```bash
cd /workspace/Qwen3-VL-Embedding
source .venv/bin/activate

python /workspace/retrieval_rerank/run_retrieval_rerank.py \
  --video-root /workspace/test \
  --query-json /workspace/ucf_crime_ar_test.json \
  --description-json /workspace/video_description.json \
  --output-dir /workspace/retrieval_rerank/outputs \
  --embedding-model Qwen/Qwen3-VL-Embedding-2B \
  --reranker-model Qwen/Qwen3-VL-Reranker-2B \
  --top-k-retrieval 50 \
  --final-mode rerank \
  --fps 1.0 \
  --max-frames 64 \
  --embedding-batch-size 6 \
  --torch-dtype bfloat16
```

## Run With nohup

```bash
cd /workspace/Qwen3-VL-Embedding
source .venv/bin/activate

nohup python /workspace/retrieval_rerank/run_retrieval_rerank.py \
  --video-root /workspace/test \
  --query-json /workspace/ucf_crime_ar_test.json \
  --description-json /workspace/video_description.json \
  --output-dir /workspace/retrieval_rerank/outputs \
  --embedding-model Qwen/Qwen3-VL-Embedding-2B \
  --reranker-model Qwen/Qwen3-VL-Reranker-2B \
  --top-k-retrieval 50 \
  --final-mode rerank \
  --fps 2.0 \
  --max-frames 64 \
  --embedding-batch-size 6 \
  --torch-dtype bfloat16 \
  > /workspace/retrieval_rerank/outputs/run.log 2>&1 &
```

Check log:

```bash
tail -f /workspace/retrieval_rerank/outputs/run.log
```

## Fusion mode

```bash
python /workspace/retrieval_rerank/run_retrieval_rerank.py \
  --final-mode fusion \
  --alpha 0.75
```

## Summarize Metrics

Write IR-style metrics such as `Hit@1`, `Hit@5`, `Hit@10`, `Hit@50`, `mean_rank`, and `median_rank`:

```bash
cd /workspace/Qwen3-VL-Embedding
source .venv/bin/activate

python /workspace/retrieval_rerank/summarize_metrics.py \
  --results-json /workspace/retrieval_rerank/outputs/final_results.json \
  --output-json /workspace/retrieval_rerank/outputs/final_results_metrics.json
```

For stage-1 retrieval metrics:

```bash
python /workspace/retrieval_rerank/summarize_metrics.py \
  --results-json /workspace/retrieval_rerank/outputs/stage1_retrieval.json \
  --output-json /workspace/retrieval_rerank/outputs/stage1_retrieval_metrics.json
```
