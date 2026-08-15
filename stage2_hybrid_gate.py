from stage0_5_policy_encoder import JournalPolicyEncoder

class Stage2RiskGate:
    """
    Stage 2: Risk-Aware Policy Gate (DTAR v3.0)
    1. Hard Integrity Gate: Drop predatory/delisted journals.
    2. Policy Conflict Score R_policy(x,j): Quantifies policy conflict risk [0.0 - 1.0].
    3. Ambiguity Bucketing (ALLOW, CONFLICT, AMBIGUOUS).
    4. Soft domain matching & user preference constraints.
    """
    def __init__(self):
        self.policy_encoder = JournalPolicyEncoder()
        
        # Predatory / Delisted blacklist for Integrity Hard Gate
        self.delisted_journals = set([
            "journal of predatory medicine",
            "fake biomedical research",
            "open access medical fraud journal"
        ])

    def evaluate_policy_conflict(self, paper_object, journal_candidate):
        """
        Computes R_policy(x,j) = sum_k w_k * c_k(x,j) and classifies into ALLOW/CONFLICT/AMBIGUOUS
        """
        policy = self.policy_encoder.extract_journal_policy(journal_candidate)
        excluded_types = policy.get("excluded_types", [])
        signals = paper_object.get("paper_signals", {})

        conflict_score = 0.0
        conflict_reasons = []

        # 1. Case Report Exclusion check
        if "case_report" in excluded_types:
            case_sig = signals.get("is_case_report", 0.0)
            if case_sig >= 0.5:
                conflict_score += 0.85 * case_sig
                conflict_reasons.append("Journal policy strictly excludes Case Reports.")
            elif case_sig >= 0.2:
                conflict_score += 0.40 * case_sig
                conflict_reasons.append("Potential case series signal conflicts with journal exclusion policy.")

        # 2. Cell Line Exclusion check
        if "cell_line_only" in excluded_types:
            cell_sig = signals.get("is_cell_line", 0.0)
            if cell_sig >= 0.5:
                conflict_score += 0.75 * cell_sig
                conflict_reasons.append("Journal excludes in vitro cell culture studies without clinical cohort.")
            elif cell_sig >= 0.2:
                conflict_score += 0.35 * cell_sig

        # 3. Animal Exclusion check
        if "animal_only" in excluded_types:
            animal_sig = signals.get("is_animal_only", 0.0)
            if animal_sig >= 0.5:
                conflict_score += 0.65 * animal_sig
                conflict_reasons.append("Journal excludes pure preclinical animal studies.")

        # Bound R_policy in [0.0, 1.0]
        r_policy = min(1.0, round(conflict_score, 3))

        # Categorize bucket
        if r_policy >= 0.65:
            bucket = "CONFLICT"
        elif r_policy >= 0.25:
            bucket = "AMBIGUOUS"
        else:
            bucket = "ALLOW"

        return r_policy, bucket, conflict_reasons, policy.get("evidence_spans", [])

    def process_candidates(self, candidate_list, paper_object, user_strict_mode=False):
        """
        Processes candidate pool (Top 50 -> Top 15 - 20) with risk scoring.
        """
        filtered_candidates = []

        for candidate in candidate_list:
            j_title = str(candidate.get('title', '')).strip()

            # --- 1. HARD INTEGRITY GATE ---
            if j_title.lower() in self.delisted_journals:
                continue

            # --- 2. POLICY CONFLICT RISK SCORING ---
            r_policy, bucket, conflict_reasons, evidence_spans = self.evaluate_policy_conflict(paper_object, candidate)

            # In strict mode, drop high CONFLICT candidates; in standard mode, retain with risk penalty
            if user_strict_mode and bucket == "CONFLICT":
                continue

            # --- 3. SOFT DOMAIN MATCH SCORING ---
            paper_domains = paper_object.get('domains', {})
            journal_domains = candidate.get('domain_flags', {})

            domain_score = 0.0
            total_weight = 0.0
            for d_name, p_score in paper_domains.items():
                j_flag = journal_domains.get(d_name, 0.0)
                domain_score += p_score * j_flag
                total_weight += p_score

            norm_domain_score = round(domain_score / max(1.0, total_weight), 3)

            # --- 4. PRE-STRATEGIC COMPOSITE SCORE ---
            dense_sim = candidate.get('normalized_dense_sim', 0.8)
            policy_compat = max(0.0, 1.0 - r_policy)
            
            # Initial screening score for Top 20 filtering
            screening_score = (0.70 * dense_sim) + (0.15 * norm_domain_score) + (0.15 * policy_compat)

            cand_copy = dict(candidate)
            cand_copy['domain_score'] = norm_domain_score
            cand_copy['policy_risk'] = r_policy
            cand_copy['policy_bucket'] = bucket
            cand_copy['policy_compatibility'] = policy_compat
            cand_copy['conflict_reasons'] = conflict_reasons
            cand_copy['evidence_spans'] = evidence_spans
            cand_copy['screening_score'] = screening_score
            cand_copy['integrity_status'] = "PASS"

            filtered_candidates.append(cand_copy)

        # Sort and select Top 20 candidates
        filtered_candidates.sort(key=lambda x: x['screening_score'], reverse=True)
        return filtered_candidates[:20]

if __name__ == "__main__":
    gate = Stage2RiskGate()
    mock_candidates = [
        {
            "title": "Neurology Clinical Cases",
            "dense_similarity_score": 0.89,
            "best_quartile": "Q1",
            "aims": "We publish case reports in neurology.",
            "scope": "Clinical neurology."
        },
        {
            "title": "Lancet Neurology",
            "dense_similarity_score": 0.92,
            "best_quartile": "Q1",
            "aims": "We do not accept case reports.",
            "scope": "Major randomized clinical trials only."
        }
    ]
    mock_paper = {
        "study_type": "Case Report",
        "paper_signals": {"is_case_report": 0.9, "is_clinical_trial": 0.05},
        "domains": {"Medicine": 0.9, "Neuroscience": 0.9}
    }
    res = gate.process_candidates(mock_candidates, mock_paper)
    for r in res:
        print(f"Journal: {r['title']} | Risk: {r['policy_risk']} | Bucket: {r['policy_bucket']}")
