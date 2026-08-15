import numpy as np

class Stage3StrategicScorer:
    """
    Stage 3 & 4: Strategic Utility Scorer (DTAR v3.0)
    Constructs multi-dimensional vector V(x,j) = [F, P, Q, I, R, U]
    Computes counterfactual-style strategic utility U(x, j | theta) balancing Fit, Policy, Quality, Risk, and User Preferences.
    """
    def __init__(self, default_preferences=None):
        self.default_preferences = default_preferences or {
            "min_quartile": "Q1",
            "fit_weight": 0.35,
            "policy_weight": 0.20,
            "quality_weight": 0.20,
            "preference_weight": 0.10,
            "integrity_weight": 0.05,
            "risk_weight": 0.10
        }

    def compute_quality_proxy(self, journal_candidate):
        """
        Computes normalized journal quality/impact proxy Q(j) in [0.0, 1.0] from SCImago Quartile, SJR, and H-Index.
        """
        quartile = str(journal_candidate.get('best_quartile', 'Q1')).upper()
        quartile_map = {'Q1': 1.0, 'Q2': 0.75, 'Q3': 0.50, 'Q4': 0.25}
        q_score = quartile_map.get(quartile, 0.6)

        try:
            sjr = float(journal_candidate.get('sjr_index', 1.0))
        except (ValueError, TypeError):
            sjr = 1.0
        # Sigmoid-normalized SJR proxy (SJR ~ 2.0 -> 0.85, SJR ~ 5.0 -> 0.98)
        norm_sjr = min(1.0, float(1.0 / (1.0 + np.exp(-0.8 * sjr + 1.2))))

        try:
            h_index = float(journal_candidate.get('h_index', 50))
        except (ValueError, TypeError):
            h_index = 50.0
        norm_hindex = min(1.0, h_index / 200.0)

        # Composite Quality Proxy
        quality_proxy = round(0.5 * q_score + 0.3 * norm_sjr + 0.2 * norm_hindex, 3)
        return quality_proxy

    def compute_preference_satisfaction(self, journal_candidate, preferences):
        """
        Computes user preference satisfaction score U_pref in [0.0, 1.0].
        """
        min_quartile = preferences.get("min_quartile", "Q2")
        q_order = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
        cand_q = str(journal_candidate.get('best_quartile', 'Q1')).upper()

        if q_order.get(cand_q, 1) >= q_order.get(min_quartile, 2):
            return 1.0
        else:
            return 0.4

    def score_candidates(self, paper_object, candidate_list, user_preferences=None):
        """
        Calculates full strategic utility for all candidates.
        """
        pref = user_preferences or self.default_preferences
        w_fit = pref.get("fit_weight", 0.35)
        w_pol = pref.get("policy_weight", 0.20)
        w_qual = pref.get("quality_weight", 0.20)
        w_pref = pref.get("preference_weight", 0.10)
        w_int = pref.get("integrity_weight", 0.05)
        w_risk = pref.get("risk_weight", 0.10)

        scored_candidates = []

        for c in candidate_list:
            # 1. Semantic Fit F
            dense_sim = c.get('normalized_dense_sim', 0.85)
            domain_score = c.get('domain_score', 0.8)
            fit_f = round(0.75 * dense_sim + 0.25 * domain_score, 3)

            # 2. Policy Compatibility P
            pol_p = round(c.get('policy_compatibility', 1.0), 3)

            # 3. Quality Proxy Q
            qual_q = self.compute_quality_proxy(c)

            # 4. Integrity I
            int_i = 1.0 if c.get('integrity_status') == "PASS" else 0.5

            # 5. Policy Risk R
            risk_r = round(c.get('policy_risk', 0.0), 3)

            # 6. Preference Satisfaction U
            pref_u = self.compute_preference_satisfaction(c, pref)

            # --- STRATEGIC UTILITY FORMULA ---
            strategic_utility = (
                w_fit * fit_f +
                w_pol * pol_p +
                w_qual * qual_q +
                w_int * int_i +
                w_pref * pref_u -
                w_risk * risk_r
            )
            strategic_utility = round(max(0.0, min(1.0, strategic_utility)), 3)

            cand_entry = dict(c)
            cand_entry['dimensions'] = {
                "fit": fit_f,
                "policy_fit": pol_p,
                "quality": qual_q,
                "preference_fit": pref_u,
                "integrity": int_i,
                "policy_risk": risk_r
            }
            cand_entry['strategic_utility'] = strategic_utility
            scored_candidates.append(cand_entry)

        # Sort by strategic utility descending
        scored_candidates.sort(key=lambda x: x['strategic_utility'], reverse=True)
        return scored_candidates

if __name__ == "__main__":
    scorer = Stage3StrategicScorer()
    sample_cands = [
        {"title": "Journal A", "dense_similarity_score": 0.90, "domain_score": 0.85, "policy_compatibility": 0.95, "policy_risk": 0.05, "best_quartile": "Q1", "sjr_index": 2.5, "integrity_status": "PASS"},
        {"title": "Journal B", "dense_similarity_score": 0.95, "domain_score": 0.80, "policy_compatibility": 0.30, "policy_risk": 0.70, "best_quartile": "Q1", "sjr_index": 3.0, "integrity_status": "PASS"}
    ]
    scored = scorer.score_candidates({}, sample_cands)
    for s in scored:
        print(f"{s['title']} -> Strategic Utility: {s['strategic_utility']} | Dims: {s['dimensions']}")
