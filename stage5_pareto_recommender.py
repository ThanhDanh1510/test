class Stage5ParetoRecommender:
    """
    Stage 5: Pareto Frontier Recommendation (DTAR v3.0)
    Filters out dominated journals across (Fit, Policy, Quality, Risk, Preference) and constructs Pareto-optimal decision options.
    """
    def __init__(self):
        pass

    def check_dominance(self, cand_a, cand_b):
        """
        Returns True if Candidate A strictly dominates Candidate B across multi-objectives.
        Objectives to maximize: Fit, Policy_fit, Quality, Preference_fit.
        Objective to minimize: Policy_risk.
        """
        dims_a = cand_a.get('dimensions', {})
        dims_b = cand_b.get('dimensions', {})

        # Vectors to maximize
        vec_a = [
            dims_a.get('fit', 0),
            dims_a.get('policy_fit', 0),
            dims_a.get('quality', 0),
            dims_a.get('preference_fit', 0),
            -dims_a.get('policy_risk', 0) # Inverted for minimization
        ]
        vec_b = [
            dims_b.get('fit', 0),
            dims_b.get('policy_fit', 0),
            dims_b.get('quality', 0),
            dims_b.get('preference_fit', 0),
            -dims_b.get('policy_risk', 0)
        ]

        not_worse = all(a >= b for a, b in zip(vec_a, vec_b))
        strictly_better = any(a > b for a, b in zip(vec_a, vec_b))

        return not_worse and strictly_better

    def compute_pareto_frontier(self, scored_candidates):
        """
        Identifies Pareto-optimal non-dominated journals and labels decision profiles.
        """
        n = len(scored_candidates)
        is_dominated = [False] * n

        for i in range(n):
            for j in range(n):
                if i != j and self.check_dominance(scored_candidates[j], scored_candidates[i]):
                    is_dominated[i] = True
                    break

        enriched_candidates = []
        for idx, cand in enumerate(scored_candidates):
            c_copy = dict(cand)
            c_copy['pareto_optimal'] = not is_dominated[idx]
            
            # Assign Decision Profile
            dims = c_copy.get('dimensions', {})
            if idx == 0:
                c_copy['decision_profile'] = "Best Overall Strategic Balance"
            elif dims.get('quality', 0) >= 0.85 and dims.get('policy_risk', 0) <= 0.2:
                c_copy['decision_profile'] = "High Prestige & Quality Pick"
            elif dims.get('policy_risk', 0) <= 0.05 and dims.get('policy_fit', 0) >= 0.9:
                c_copy['decision_profile'] = "Safest Policy & Scope Bet"
            elif dims.get('fit', 0) >= 0.90:
                c_copy['decision_profile'] = "Highest Semantic Precision"
            else:
                c_copy['decision_profile'] = "Alternative Viable Option"

            enriched_candidates.append(c_copy)

        return enriched_candidates

if __name__ == "__main__":
    recommender = Stage5ParetoRecommender()
    cands = [
        {"title": "J1", "dimensions": {"fit": 0.9, "policy_fit": 0.9, "quality": 0.9, "preference_fit": 1.0, "policy_risk": 0.1}},
        {"title": "J2", "dimensions": {"fit": 0.7, "policy_fit": 0.7, "quality": 0.7, "preference_fit": 0.8, "policy_risk": 0.3}},
        {"title": "J3", "dimensions": {"fit": 0.6, "policy_fit": 0.95, "quality": 0.95, "preference_fit": 1.0, "policy_risk": 0.05}}
    ]
    frontier = recommender.compute_pareto_frontier(cands)
    for f in frontier:
        print(f"{f['title']} | Pareto Optimal: {f['pareto_optimal']} | Profile: {f['decision_profile']}")
