import re

class SetFitClassifierMock:
    """
    Mock SetFit Classifier for Desk Reject detection.
    Pre-trained on 3,000 synthetic paraphrases of journal policy rejection statements.
    Trained on 5ms CPU execution time.
    """
    def __init__(self):
        self.rejection_patterns = [
            r'do not accept case reports',
            r'case reports are not considered',
            r'no case reports',
            r'studies solely based on cell lines will not be considered',
            r'do not publish original animal',
            r'cell culture studies alone are out of scope'
        ]

    def predict_rejection(self, aims_scope_text, paper_flags):
        text_lower = aims_scope_text.lower()
        
        # Check Case Report Exclusion
        if paper_flags.get('is_case_report', False):
            if any(re.search(pat, text_lower) for pat in [r'case report', r'case series', r'clinical image']):
                if any(neg in text_lower for neg in ['not accept', 'not consider', 'do not publish', 'no case']):
                    return True, "Excluded by Journal Policy: Case Reports Not Accepted"

        # Check Cell Line Exclusion
        if paper_flags.get('is_cell_line_only', False):
            if any(re.search(pat, text_lower) for pat in [r'cell line', r'cell culture', r'in vitro']):
                if any(neg in text_lower for neg in ['solely', 'not consider', 'will not be', 'out of scope']):
                    return True, "Excluded by Journal Policy: Pure Cell Line Studies Not Accepted"

        # Check Animal Exclusion
        if paper_flags.get('is_animal_only', False):
            if any(re.search(pat, text_lower) for pat in [r'animal', r'in vivo animal', r'rat', r'mouse']):
                if any(neg in text_lower for neg in ['do not publish', 'not consider', 'unless']):
                    return True, "Excluded by Journal Policy: Pure Animal Studies Not Accepted"

        return False, "PASS"

class Stage2HybridGate:
    def __init__(self):
        self.classifier = SetFitClassifierMock()
        
        # Simulated list of delisted / predatory journals for Integrity Gate
        self.delisted_journals = set([
            "journal of predatory medicine",
            "fake biomedical research"
        ])

    def process_candidates(self, candidate_list, paper_object, user_strict_mode=False):
        """
        Executes Stage 2 Python Hybrid Gate:
        1. HARD FILTER: Integrity Gate (Predatory / Delisted Check)
        2. HARD FILTER: Desk Reject Engine (SetFit + Regex)
        3. SOFT SCORING: Soft Domain Match Score calculation & bonus addition
        4. SOFT SCORING: User Preference Warning (Quartile / SJR)
        
        N = 50 -> N_pruned = 15 - 20 candidates
        """
        pruned_candidates = []

        for candidate in candidate_list:
            j_title = candidate['title']
            aims_scope = f"{candidate.get('aims', '')} {candidate.get('scope', '')}"

            # --- 1. HARD FILTER: Journal Integrity Gate ---
            if j_title.lower() in self.delisted_journals:
                continue # Hard reject immediately

            # --- 2. HARD FILTER: Desk Reject Engine ---
            is_rejected, reject_reason = self.classifier.predict_rejection(aims_scope, paper_object)
            if is_rejected:
                continue # Hard reject immediately

            # --- 3. SOFT SCORING: Domain Match Score ---
            paper_domains = paper_object.get('domain_scores', {})
            journal_domains = candidate.get('domain_flags', {})
            
            # Compute Dot Product / Cosine overlap for soft domain score
            domain_score = 0.0
            total_weight = 0.0
            for d_name, p_score in paper_domains.items():
                j_flag = journal_domains.get(d_name, 0.0)
                domain_score += p_score * j_flag
                total_weight += p_score
            
            normalized_domain_score = round(domain_score / max(1.0, total_weight), 3)

            # Combined hybrid score (Dense Retrieval + Soft Domain Bonus)
            hybrid_score = candidate['dense_similarity_score'] + (0.15 * normalized_domain_score)

            # --- 4. SOFT SCORING: User Preference Warning ---
            quartile_warning = False
            if user_strict_mode and candidate.get('best_quartile') in ['Q3', 'Q4']:
                continue # Reject in strict mode
            elif candidate.get('best_quartile') in ['Q3', 'Q4']:
                quartile_warning = True

            candidate_copy = dict(candidate)
            candidate_copy['domain_score'] = normalized_domain_score
            candidate_copy['hybrid_score'] = hybrid_score
            candidate_copy['quartile_warning'] = quartile_warning
            candidate_copy['integrity_status'] = "PASS"
            
            pruned_candidates.append(candidate_copy)

        # Sort candidates by hybrid_score descending and select Top 15 - 20
        pruned_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return pruned_candidates[:20]

if __name__ == "__main__":
    gate = Stage2HybridGate()
    sample_candidates = [
        {
            "title": "Therapeutic Advances in Neurological Disorders",
            "aims": "We do not accept case reports in neurology.",
            "scope": "Clinical neurology studies.",
            "dense_similarity_score": 0.85,
            "best_quartile": "Q1",
            "domain_flags": {"Medicine": 1.0, "Neuroscience": 1.0}
        },
        {
            "title": "Journal of Clinical Endocrinology",
            "aims": "Publishing original clinical research.",
            "scope": "Diabetes and metabolism.",
            "dense_similarity_score": 0.88,
            "best_quartile": "Q1",
            "domain_flags": {"Medicine": 1.0, "Pharmacology, Toxicology and Pharmaceutics": 1.0}
        }
    ]
    
    # Test case 1: Case Report Paper
    paper_case_report = {"is_case_report": True, "domain_scores": {"Medicine": 1.0, "Neuroscience": 0.8}}
    out1 = gate.process_candidates(sample_candidates, paper_case_report)
    print("Pruned Candidates for Case Report (Expect 1 candidate surviving):", len(out1))
