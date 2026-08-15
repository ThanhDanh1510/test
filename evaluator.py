import numpy as np
import time

def evaluate_pipeline(pipeline, test_papers_df, top_k_list=[1, 3, 5, 10]):
    """
    Evaluates End-to-End DTAR-Slim v2.1 Pipeline metrics:
    - Recall@50 (Stage 1 Retriever)
    - Accuracy@K (K=1, 3, 5, 10)
    - NDCG@K (K=5, 10)
    - Kendall's Tau Average
    - Latency (P50, P95)
    """
    print(f"\n=======================================================")
    print(f"[Evaluator] Starting Evaluation on {len(test_papers_df)} papers...")
    print(f"=======================================================\n")

    stage1_recalls_50 = []
    acc_hits = {k: 0 for k in top_k_list}
    ndcg_scores = {5: [], 10: []}
    tau_correlations = []
    latencies = []

    start_eval_time = time.time()

    for idx, row in test_papers_df.iterrows():
        t0 = time.time()
        
        target_label = int(row['Label'])
        target_journal_title = pipeline.loader.label_to_journal.get(target_label, {}).get('title', '').lower().strip()
        paper_input = {
            "title": str(row['Title']),
            "abstract": str(row['Abstract']),
            "keywords": str(row.get('Keywords', ''))
        }

        # Run End-to-End Pipeline
        res_output, stage1_top50 = pipeline.run(paper_input, return_stage1=True)
        
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0) # Ms

        # 1. Stage 1 Recall@50
        s1_ids = [int(c.get('journal_id', -1)) for c in stage1_top50]
        s1_titles = [c.get('title', '').lower().strip() for c in stage1_top50]
        is_in_stage1 = (target_label in s1_ids) or (target_journal_title.lower().strip() in s1_titles)
        stage1_recalls_50.append(1.0 if is_in_stage1 else 0.0)

        # 2. Accuracy@K, Acceptable Sister Set Accuracy & NDCG@K
        hit_rank = None
        acceptable_hit_rank = None
        target_cats = pipeline.loader.label_to_journal.get(target_label, {}).get('categories', [])
        target_cats_set = set([str(c).lower().strip() for c in target_cats])

        for rank, item in enumerate(res_output, 1):
            item_title = item.get('journal_title', '').lower().strip()
            item_id = pipeline.loader.journal_to_label.get(item_title, -1)
            
            # Strict exact match
            if (item_id == target_label or item_title == target_journal_title) and hit_rank is None:
                hit_rank = rank
            
            # Acceptable Sister Set match (same major category and Q1 quartile)
            item_cats = item.get('categories', [])
            item_cats_set = set([str(c).lower().strip() for c in item_cats])
            if (item_id == target_label or bool(target_cats_set.intersection(item_cats_set))) and item.get('best_quartile') == 'Q1' and acceptable_hit_rank is None:
                acceptable_hit_rank = rank

        for k in top_k_list:
            if hit_rank and hit_rank <= k:
                acc_hits[k] += 1

        # Calculate NDCG@5 & NDCG@10
        for k in [5, 10]:
            if hit_rank and hit_rank <= k:
                ndcg_scores[k].append(1.0 / np.log2(hit_rank + 1))
            else:
                ndcg_scores[k].append(0.0)

        # 3. Kendall's Tau Tracking
        if len(res_output) > 0:
            tau_correlations.append(res_output[0]['confidence_flags']['kendall_tau'])

    total_time = time.time() - start_eval_time
    total_samples = len(test_papers_df)

    # Compile Summary Report
    report = {
        "samples_evaluated": total_samples,
        "total_eval_time_sec": round(total_time, 2),
        "stage1_recall_50": round(float(np.mean(stage1_recalls_50)), 4),
        "accuracy": {f"Top-{k}": round(acc_hits[k] / total_samples, 4) for k in top_k_list},
        "ndcg": {f"NDCG@{k}": round(float(np.mean(ndcg_scores[k])), 4) for k in [5, 10]},
        "kendall_tau_mean": round(float(np.mean(tau_correlations)), 4),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2)
        }
    }

    print("\n---------------- EVALUATION RESULTS (DTAR-Slim v2.1) ----------------")
    print(f"Evaluated Samples  : {report['samples_evaluated']}")
    print(f"Stage 1 Recall@50  : {report['stage1_recall_50'] * 100:.2f}%")
    for k_str, acc_val in report['accuracy'].items():
        print(f"Accuracy ({k_str})    : {acc_val * 100:.2f}%")
    for k_str, ndcg_val in report['ndcg'].items():
        print(f"{k_str}              : {ndcg_val:.4f}")
    print(f"Kendall's Tau Mean : {report['kendall_tau_mean']}")
    print(f"Latency P50        : {report['latency_ms']['p50']} ms")
    print(f"Latency P95        : {report['latency_ms']['p95']} ms")
    print("---------------------------------------------------------------------\n")

    return report
