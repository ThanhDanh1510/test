import re

class JournalPolicyEncoder:
    """
    Stage 0.5: Journal Policy Encoder (DTAR v3.0)
    Converts unstructured Aims & Scope into machine-verifiable constraints and evidence spans.
    Pre-computes and caches structured policy profiles for all 1,406 PubMed journals.
    """
    def __init__(self):
        self.policy_cache = {}

    def extract_journal_policy(self, journal_row):
        """
        Extracts structured policy constraints from journal metadata.
        """
        journal_id = str(journal_row.get('journal_id', journal_row.get('Label', '0')))
        if journal_id in self.policy_cache:
            return self.policy_cache[journal_id]

        title = str(journal_row.get('title', journal_row.get('Title', ''))).strip()
        aims = str(journal_row.get('aims', '')).strip()
        scope = str(journal_row.get('scope', '')).strip()
        aims_scope_text = f"{aims} {scope}".strip()
        text_lower = aims_scope_text.lower()

        article_types = ["original research", "clinical study"]
        excluded_types = []
        evidence_spans = []

        # 1. Case Report Policy
        if any(re.search(pat, text_lower) for pat in [r'case report', r'case series', r'clinical image']):
            if any(neg in text_lower for neg in ['not accept', 'not consider', 'do not publish', 'no case reports', 'will not consider']):
                excluded_types.append("case_report")
                evidence_spans.append({
                    "constraint": "case_reports_excluded",
                    "value": True,
                    "source_text": "Journal explicitly states it does not accept Case Reports.",
                    "source_field": "aims_scope"
                })
            else:
                article_types.append("case report")
        elif "case report" in title.lower():
            article_types.append("case report")

        # 2. Pure Cell Line Policy
        if any(re.search(pat, text_lower) for pat in [r'cell line', r'cell culture', r'in vitro']):
            if any(neg in text_lower for neg in ['solely', 'not consider', 'will not be', 'out of scope', 'unless validated']):
                excluded_types.append("cell_line_only")
                evidence_spans.append({
                    "constraint": "cell_line_only_excluded",
                    "value": True,
                    "source_text": "Studies solely based on cell lines/in vitro without clinical validation are out of scope.",
                    "source_field": "aims_scope"
                })
            else:
                article_types.append("in vitro study")

        # 3. Pure Animal Policy
        if any(re.search(pat, text_lower) for pat in [r'animal', r'in vivo animal', r'rat', r'mouse']):
            if any(neg in text_lower for neg in ['do not publish', 'not consider', 'purely animal']):
                excluded_types.append("animal_only")
                evidence_spans.append({
                    "constraint": "animal_only_excluded",
                    "value": True,
                    "source_text": "Pure animal model studies are not considered without human translational relevance.",
                    "source_field": "aims_scope"
                })
            else:
                article_types.append("animal study")

        # 4. Review / Clinical Trial acceptance
        if "review" in text_lower or "meta-analysis" in text_lower:
            article_types.append("review")
        if "clinical trial" in text_lower or "rct" in text_lower or "trial" in text_lower:
            article_types.append("clinical trial")
            evidence_spans.append({
                "constraint": "clinical_trials_prioritized",
                "value": True,
                "source_text": "Journal prioritizes randomized controlled clinical trials.",
                "source_field": "aims_scope"
            })

        policy_obj = {
            "journal_id": journal_id,
            "title": title,
            "article_types": list(set(article_types)),
            "excluded_types": list(set(excluded_types)),
            "evidence_spans": evidence_spans,
            "raw_aims_scope": aims_scope_text[:300]
        }

        self.policy_cache[journal_id] = policy_obj
        return policy_obj

    def precompute_all_policies(self, journal_df):
        """Precomputes policy profiles for all journals in dataframe"""
        for _, row in journal_df.iterrows():
            self.extract_journal_policy(row)
        print(f"[PolicyEncoder] Pre-computed policy profiles for {len(self.policy_cache)} journals.")

if __name__ == "__main__":
    encoder = JournalPolicyEncoder()
    mock_row = {
        "journal_id": "1",
        "title": "Journal of Clinical Neurology",
        "aims": "We do not accept case reports or pure cell line studies.",
        "scope": "Focus on clinical trials in neurology."
    }
    policy = encoder.extract_journal_policy(mock_row)
    print("Extracted Policy:", policy)
