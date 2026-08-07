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
        s1_ids = [c['journal_id'] for c in stage1_top50]
        stage1_recalls_50.append(1.0 if target_label in s1_ids else 0.0)

        # 2. Accuracy@K & NDCG@K
        predicted_journal_titles = [item['journal_title'].lower() for item in res_output]
        target_journal_title = pipeline.loader.label_to_journal.get(target_label, {}).get('title', '').lower()

        # Track hit position
        hit_rank = None
        for rank, item in enumerate(res_output, 1):
            if pipeline.loader.journal_to_label.get(item['journal_title'].lower()) == target_label or item['journal_title'].lower() == target_journal_title:
                hit_rank = rank
                break

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

    print("\n---------------- EVALUATION RESULTS ----------------")
    print(f"Evaluated Samples  : {report['samples_evaluated']}")
    print(f"Stage 1 Recall@50  : {report['stage1_recall_50'] * 100:.2f}%")
    for k_str, acc_val in report['accuracy'].items():
        print(f"Accuracy ({k_str})    : {acc_val * 100:.2f}%")
    for k_str, ndcg_val in report['ndcg'].items():
        print(f"{k_str}              : {ndcg_val:.4f}")
    print(f"Kendall's Tau Mean : {report['kendall_tau_mean']}")
    print(f"Latency P50        : {report['latency_ms']['p50']} ms")
    print(f"Latency P95        : {report['latency_ms']['p95']} ms")
    print("----------------------------------------------------\n")

    return report
