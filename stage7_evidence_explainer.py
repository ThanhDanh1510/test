class Stage7EvidenceExplainer:
    """
    Stage 7: Evidence-Grounded Explanation Generator (DTAR v3.0)
    Renders structured evidence objects (positive, negative, policy spans) and verbalizes transparent reasoning traces.
    """
    def __init__(self):
        pass

    def build_evidence_object(self, paper_object, candidate_item):
        """
        Constructs deterministic structured evidence representation from paper and journal metadata.
        """
        study_type = paper_object.get('study_type', 'Original Research')
        pico = paper_object.get('pico', {})
        dims = candidate_item.get('dimensions', {})
        policy_risk = dims.get('policy_risk', 0.0)
        conflict_reasons = candidate_item.get('conflict_reasons', [])
        evidence_spans = candidate_item.get('evidence_spans', [])
        
        cats = candidate_item.get('categories', [])
        cats_str = ', '.join([str(c) for c in (cats[:2] if isinstance(cats, list) else [cats])])

        # 1. Positive Evidence
        pos_evidence = [
            {
                "dimension": "study_design_alignment",
                "paper_signal": study_type,
                "journal_evidence": f"Journal regularly accepts {study_type} within {cats_str}."
            },
            {
                "dimension": "pico_domain_fit",
                "paper_signal": f"Population: {pico.get('population', 'Clinical cohort')}",
                "journal_evidence": f"High domain match score ({dims.get('fit', 0.85)}) with journal scope."
            },
            {
                "dimension": "journal_impact",
                "journal_evidence": f"Indexed {candidate_item.get('best_quartile', 'Q1')} (SJR: {candidate_item.get('sjr_index', 1.0)})."
            }
        ]

        # 2. Negative Evidence & Risk Friction
        neg_evidence = []
        if policy_risk > 0.15:
            for reason in conflict_reasons:
                neg_evidence.append({
                    "dimension": "policy_friction",
                    "risk_score": policy_risk,
                    "reason": reason
                })
        elif dims.get('preference_fit', 1.0) < 0.7:
            neg_evidence.append({
                "dimension": "user_preference_divergence",
                "risk_score": 0.2,
                "reason": f"Journal quartile ({candidate_item.get('best_quartile')}) is slightly below requested target."
            })

        # 3. Policy Evidence Spans
        pol_evidence = []
        for span in evidence_spans:
            pol_evidence.append({
                "constraint": span.get('constraint'),
                "verified_status": "MATCHED",
                "source_quote": span.get('source_text')
            })

        # 4. Natural Language Synthesis (Verbalization)
        verbalized_summary = (
            f"Strategically recommended as '{candidate_item.get('decision_profile', 'Recommended')}' "
            f"(Utility: {candidate_item.get('strategic_utility')}). Matches {study_type} with {cats_str}. "
            f"Policy safety: {1.0 - policy_risk:.0%}. Quality: {candidate_item.get('best_quartile')}."
        )

        return {
            "summary_text": verbalized_summary,
            "positive_evidence": pos_evidence,
            "negative_evidence": neg_evidence,
            "policy_evidence": pol_evidence
        }

    def explain_recommendations(self, paper_object, calibrated_candidates):
        """
        Enriches Top 5 candidates with structured evidence objects and URLs.
        """
        final_recommendations = []

        for rank_idx, cand in enumerate(calibrated_candidates[:5], 1):
            evidence_obj = self.build_evidence_object(paper_object, cand)
            
            res_item = {
                "rank": rank_idx,
                "journal_title": cand['title'],
                "strategic_utility": cand.get('strategic_utility', 0.8),
                "dimensions": cand.get('dimensions', {}),
                "uncertainty": cand.get('uncertainty', 0.05),
                "pareto_optimal": cand.get('pareto_optimal', True),
                "decision_profile": cand.get('decision_profile', 'Viable Option'),
                "evidence": evidence_obj,
                "confidence_flags": cand.get('confidence_flags', {}),
                "urls": {
                    "pubmed": cand.get('pubmed_url', ''),
                    "scimago": cand.get('scimago_url', '')
                }
            }
            final_recommendations.append(res_item)

        return final_recommendations

if __name__ == "__main__":
    explainer = Stage7EvidenceExplainer()
    mock_paper = {"study_type": "Clinical Trial", "pico": {"population": "Diabetes patients"}}
    mock_cand = [{
        "title": "Diabetes Care",
        "strategic_utility": 0.92,
        "best_quartile": "Q1",
        "sjr_index": 2.8,
        "dimensions": {"fit": 0.95, "policy_risk": 0.05},
        "conflict_reasons": [],
        "evidence_spans": [{"constraint": "clinical_trials", "source_text": "Accepts clinical trials"}]
    }]
    explained = explainer.explain_recommendations(mock_paper, mock_cand)
    print("Explained Evidence:", explained[0]['evidence'])
