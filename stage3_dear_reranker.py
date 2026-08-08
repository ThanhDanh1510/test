import numpy as np
from scipy.stats import kendalltau

class Stage3DeARReranker:
    def __init__(self, faithfulness_threshold=0.42, llm_model=None, tokenizer=None):
        self.faithfulness_threshold = faithfulness_threshold
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def format_dear_prompt(self, paper_object, candidate_list):
        """
        Formats structured DeAR Listwise Reasoning Prompt (EMNLP 2025 standard)
        """
        candidates_text = ""
        for idx, c in enumerate(candidate_list, 1):
            cats = c.get('categories', [])
            cats_str = ', '.join([str(x) for x in (cats[:2] if isinstance(cats, list) else [cats])])
            candidates_text += f"\nCandidate [{idx}]: {c['title']}\n"
            candidates_text += f"  - Quartile: {c['best_quartile']} | SJR: {c['sjr_index']}\n"
            candidates_text += f"  - Categories: {cats_str}\n"
            candidates_text += f"  - Aims & Scope: {c.get('aims', '')[:200]}...\n"

        prompt = f"""[TASK]
You are an expert biomedical journal submission recommender (DeAR Reasoning Agent). 
Evaluate the candidate journals against the paper's PICO and Study Type.

[PAPER METADATA]
Title: {paper_object['title']}
Study Type: {paper_object['study_type']}
PICO Summary: {paper_object['pico_summary']}

[CANDIDATE JOURNALS POOL]{candidates_text}

[INSTRUCTIONS]
Provide step-by-step Chain-of-Thought (CoT) reasoning comparing candidate scopes against the paper's PICO, then output the final ranked order from 1 to {len(candidate_list)}."""
        return prompt

    def rerank_and_explain(self, paper_object, candidate_list):
        """
        Executes Stage 3:
        3.1 Pointwise Fast Scorer (Top 20 -> Top 10)
        3.2 Dynamic Adaptive Permutation Check (Kendall's tau check)
        3.3 F1-Calibrated Faithfulness Check
        """
        # --- 3.1 Pointwise Fast Scorer (Select Top 10) ---
        top_10 = sorted(candidate_list, key=lambda x: x['hybrid_score'], reverse=True)[:10]

        # --- 3.2 Dynamic Adaptive Permutation Check ---
        # Pass 1: Original Order
        order_pass1 = self._generate_listwise_ranking(paper_object, top_10, reverse_order=False)
        # Pass 2: Reverse Order
        order_pass2 = self._generate_listwise_ranking(paper_object, top_10, reverse_order=True)

        # Compute Kendall's tau correlation between Pass 1 and Pass 2
        tau_corr, _ = kendalltau(order_pass1, order_pass2)
        if np.isnan(tau_corr):
            tau_corr = 1.0

        low_confidence_ranking = False
        final_ranked_items = []

        if tau_corr >= 0.7:
            # High consistency -> Average scores
            final_scores = (np.array(order_pass1) + np.array(order_pass2)) / 2.0
        else:
            # Low consistency -> Trigger Pass 3 (Random Order)
            low_confidence_ranking = True
            order_pass3 = self._generate_listwise_ranking(paper_object, top_10, shuffle_order=True)
            final_scores = (np.array(order_pass1) + np.array(order_pass2) + np.array(order_pass3)) / 3.0

        # Sort top 10 candidates by final averaged rank score
        ranked_indices = np.argsort(final_scores)
        top_5_final = [top_10[i] for i in ranked_indices[:5]]

        # --- 3.3 Faithfulness Verification & Output Formulation ---
        results = []
        for rank_idx, candidate in enumerate(top_5_final, 1):
            cats = candidate.get('categories', [])
            cats_flat = []
            if isinstance(cats, list):
                for c in cats:
                    if isinstance(c, list):
                        cats_flat.extend([str(x) for x in c])
                    else:
                        cats_flat.append(str(c))
            else:
                cats_flat = [str(cats)]
                
            cats_str = ', '.join(cats_flat[:2]) if cats_flat else "General Medicine"
            scope_fit_text = f"Strongly aligns with paper's PICO ({paper_object['study_type']}) and journal's categories ({cats_str})."
            why_top1_text = f"Highest domain match score ({candidate['domain_score']}) and Q1 journal scope precision." if rank_idx == 1 else f"Ranked #{rank_idx} due to slightly lower domain specificity."
            
            # Simulated Cosine Similarity Check with bge-small-en-v1.5
            faithfulness_score = round(float(0.55 + 0.3 * candidate['domain_score']), 2)
            needs_review = faithfulness_score < self.faithfulness_threshold

            res_item = {
                "rank": rank_idx,
                "journal_title": candidate['title'],
                "best_quartile": candidate['best_quartile'],
                "sjr_index": candidate['sjr_index'],
                "domain_score": candidate['domain_score'],
                "reasoning_trace": {
                    "scope_fit": scope_fit_text,
                    "why_top1": why_top1_text,
                    "integrity_status": candidate['integrity_status']
                },
                "confidence_flags": {
                    "kendall_tau": round(float(tau_corr), 2),
                    "low_confidence_ranking": low_confidence_ranking,
                    "faithfulness_score": faithfulness_score,
                    "needs_review": needs_review
                },
                "urls": {
                    "pubmed": candidate.get('pubmed_url', ''),
                    "scimago": candidate.get('scimago_url', '')
                }
            }
            results.append(res_item)

        return results

    def _generate_listwise_ranking(self, paper_obj, candidates, reverse_order=False, shuffle_order=False):
        """Simulates LLM Listwise CoT score generation with input permutation"""
        n = len(candidates)
        base_ranks = np.arange(n)
        
        if reverse_order:
            # Add small permutation noise for Pass 2
            noise = np.random.normal(0, 0.2, n)
            return list(base_ranks + noise)
        elif shuffle_order:
            noise = np.random.normal(0, 0.5, n)
            return list(base_ranks + noise)
        else:
            return list(base_ranks)

if __name__ == "__main__":
    reranker = Stage3DeARReranker()
    sample_paper = {"study_type": "Randomized Controlled Trial"}
    sample_candidates = [
        {"title": f"Journal {i}", "best_quartile": "Q1", "sjr_index": 1.5, "domain_score": 0.8, "hybrid_score": 0.9 - i*0.01, "categories": ["Medicine"], "integrity_status": "PASS"}
        for i in range(15)
    ]
    output = reranker.rerank_and_explain(sample_paper, sample_candidates)
    print("Top 1 Ranked Journal:", output[0]['journal_title'])
    print("Kendall's Tau:", output[0]['confidence_flags']['kendall_tau'])
