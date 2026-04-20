import json
import statistics
from pathlib import Path
import argparse

def ground_truth_matches(candidate_relative_path, ground_truth_video):
    if not ground_truth_video:
        return False
    gt = str(Path(ground_truth_video))
    candidate = str(candidate_relative_path)
    if candidate == gt:
        return True
    return candidate.endswith("/" + gt)

def hit_at(ranks, k, total):
    if total == 0:
        return 0.0
    return sum(1 for rank in ranks if rank <= k) / total

def process_stage2_data(data, alpha):
    matched_ranks = []
    
    n_queries = 0
    for query in data:
        gt = query.get("ground_truth_video")
        if not gt:
            continue
        n_queries += 1
            
        candidates = query.get("candidates", [])
        
        # Calculate final scores
        for c in candidates:
            v_score = c.get("visual_score", 0.0)
            r_score = c.get("rerank_score", 0.0)
            c["final_score_tmp"] = alpha * v_score + (1.0 - alpha) * r_score
            
        # Sort candidates descending based on final_score
        sorted_candidates = sorted(candidates, key=lambda x: x["final_score_tmp"], reverse=True)
        
        # Find rank
        rank = None
        for i, c in enumerate(sorted_candidates, start=1):
            if ground_truth_matches(c["relative_path"], gt):
                rank = i
                break
                
        if rank is not None:
            matched_ranks.append(rank)

    h1 = hit_at(matched_ranks, 1, n_queries)
    h5 = hit_at(matched_ranks, 5, n_queries)
    h10 = hit_at(matched_ranks, 10, n_queries)
    mean_r = statistics.mean(matched_ranks) if matched_ranks else 0.0
    
    return {"hit_at_1": h1, "hit_at_5": h5, "hit_at_10": h10, "mean_rank": mean_r}

def main():
    parser = argparse.ArgumentParser(description="Generate fusion summary table")
    parser.add_argument("--input", type=str, default="output_fusion/stage2_rerank.json", help="Path to stage2_rerank.json")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    configs = [
        {"name": "Chỉ dùng Rerank", "alpha": 0.00},
        {"name": "Chỉ dùng Visual", "alpha": 1.00},
        {"name": "Fusion (Thấp)", "alpha": 0.25},
        {"name": "Fusion (Cân bằng)", "alpha": 0.50},
        {"name": "Fusion (Tối ưu nhất)", "alpha": 0.75},
        {"name": "Fusion (Cao)", "alpha": 0.80},
    ]
    
    print(f"| {'Cấu hình':<25} | {'Alpha':<5} | {'Hit@1':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'Mean Rank':<9} |")
    print(f"|{'-'*27}|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|")
    
    for cfg in configs:
        res = process_stage2_data(data, cfg["alpha"])
        name = cfg['name']
        alpha_str = f"{cfg['alpha']:.2f}"
        h1 = f"{res['hit_at_1']*100:.2f}%"
        h5 = f"{res['hit_at_5']*100:.2f}%"
        h10 = f"{res['hit_at_10']*100:.2f}%"
        mr = f"{res['mean_rank']:.2f}"
        
        print(f"| {name:<25} | {alpha_str:<5} | {h1:<8} | {h5:<8} | {h10:<8} | {mr:<9} |")

if __name__ == '__main__':
    main()
