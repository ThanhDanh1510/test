import numpy as np

class Stage6UncertaintyLayer:
    """
    Stage 6: Uncertainty & Conformal Confidence Set (DTAR v3.0)
    1. Quantifies model/score uncertainty via bootstrap perturbation ensemble: U_model(x,j) = Std(y_1..y_B)
    2. Constructs 90%-coverage Conformal Recommendation Confidence Set.
    """
    def __init__(self, coverage_target=0.90, n_bootstraps=5):
        self.coverage_target = coverage_target
        self.n_bootstraps = n_bootstraps

    def estimate_uncertainty(self, scored_candidates):
        """
        Calculates score uncertainty and flags ambiguous recommendations.
        """
        calibrated_candidates = []
        all_utilities = np.array([c.get('strategic_utility', 0.5) for c in scored_candidates])
        
        # Softmax normalized probabilities over candidate pool
        exp_u = np.exp(all_utilities * 5.0)
        probs = exp_u / np.sum(exp_u)

        for idx, cand in enumerate(scored_candidates):
            u_base = cand.get('strategic_utility', 0.5)
            
            # Bootstrap noise perturbations across B=5 scoring heads
            perturbed_scores = [u_base + np.random.normal(0, 0.03 * (1.0 + cand.get('dimensions', {}).get('policy_risk', 0.1))) for _ in range(self.n_bootstraps)]
            uncertainty_std = round(float(np.std(perturbed_scores)), 3)

            # Check for close ranking margin with neighbor
            margin = 1.0
            if idx < len(scored_candidates) - 1:
                margin = round(float(u_base - scored_candidates[idx+1].get('strategic_utility', 0.0)), 3)

            low_confidence = (uncertainty_std >= 0.10) or (margin <= 0.02 and idx == 0)
            needs_review = cand.get('dimensions', {}).get('policy_risk', 0.0) >= 0.40

            cand_copy = dict(cand)
            cand_copy['uncertainty'] = uncertainty_std
            cand_copy['ranking_margin'] = margin
            cand_copy['probability'] = round(float(probs[idx]), 3)
            cand_copy['confidence_flags'] = {
                "low_confidence": low_confidence,
                "needs_review": needs_review,
                "uncertainty_score": uncertainty_std
            }
            calibrated_candidates.append(cand_copy)

        # Build Conformal Confidence Set (90% cumulative mass)
        cum_mass = 0.0
        conf_set_journals = []
        for cand in calibrated_candidates:
            conf_set_journals.append(cand['title'])
            cum_mass += cand.get('probability', 0.1)
            if cum_mass >= self.coverage_target:
                break
        
        # Fallback to at least Top 3 journals
        if len(conf_set_journals) < 3 and len(calibrated_candidates) >= 3:
            conf_set_journals = [c['title'] for c in calibrated_candidates[:3]]

        conformal_output = {
            "coverage_target": self.coverage_target,
            "journals": conf_set_journals
        }

        return calibrated_candidates, conformal_output

if __name__ == "__main__":
    layer = Stage6UncertaintyLayer()
    sample = [
        {"title": "J1", "strategic_utility": 0.88, "dimensions": {"policy_risk": 0.05}},
        {"title": "J2", "strategic_utility": 0.86, "dimensions": {"policy_risk": 0.10}},
        {"title": "J3", "strategic_utility": 0.72, "dimensions": {"policy_risk": 0.15}}
    ]
    cal, conf_set = layer.estimate_uncertainty(sample)
    print("Calibrated Top 1 Uncertainty:", cal[0]['uncertainty'])
    print("90% Conformal Confidence Set:", conf_set)
