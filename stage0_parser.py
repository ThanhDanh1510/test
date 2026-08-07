import re

class Stage0Parser:
    def __init__(self, mode="rule_fastpass"):
        """
        mode: 'rule_fastpass' (rule/regex fast-pass for high throughput)
              'llm_structured' (full LLM parsing)
        """
        self.mode = mode
        
        # 9 Major Domain keywords for heuristic soft scoring
        self.domain_keywords = {
            'Medicine': ['clinical', 'patient', 'disease', 'hospital', 'therapy', 'treatment', 'surgery', 'syndrome'],
            'Neuroscience': ['brain', 'neuro', 'cortex', 'synaptic', 'neuron', 'cognitive', 'parkinson', 'alzheimer'],
            'Immunology and Microbiology': ['immune', 'antibody', 'virus', 'viral', 'bacterial', 'pathogen', 'infection', 'cytokine'],
            'Biochemistry, Genetics and Molecular Biology': ['gene', 'protein', 'dna', 'rna', 'molecular', 'expression', 'cellular', 'genome'],
            'Pharmacology, Toxicology and Pharmaceutics': ['drug', 'pharmacology', 'dose', 'toxicity', 'inhibitor', 'receptor', 'compound'],
            'Health Professions': ['rehabilitation', 'occupational', 'physical therapy', 'radiology', 'health care'],
            'Nursing': ['nursing', 'nurse', 'caregiving', 'patient care', 'palliative'],
            'Psychology': ['behavior', 'psychological', 'depression', 'anxiety', 'mental health', 'psychiatry'],
            'Dentistry': ['dental', 'tooth', 'oral', 'periodontal', 'dentistry', 'orthodontic']
        }

    def parse_paper(self, title, abstract, keywords=""):
        """
        Extracts structured paper metadata:
        - PICO summary
        - Study Type & Binary Flags (is_case_report, is_cell_line_only, is_animal_only)
        - Soft Domain Scores (0.0 - 1.0)
        """
        full_text = f"{title} {abstract} {keywords}".lower()

        # 1. Study Type Extraction & Binary Flags (Rule Fast-Pass)
        is_case_report = bool(re.search(r'\b(case report|case series|clinical image|photo essay)\b', full_text))
        is_cell_line_only = bool(re.search(r'\b(in vitro|cell line|cell culture|hela|hek293)\b', full_text)) and not bool(re.search(r'\b(patient|clinical trial|human subjects)\b', full_text))
        is_animal_only = bool(re.search(r'\b(rat|mice|mouse|canine|porcine|in vivo animal)\b', full_text)) and not bool(re.search(r'\b(patient|human|clinical)\b', full_text))

        if is_case_report:
            study_type = "Case Report"
        elif "randomized controlled trial" in full_text or "rct" in full_text:
            study_type = "Randomized Controlled Trial"
        elif "systematic review" in full_text or "meta-analysis" in full_text:
            study_type = "Systematic Review / Meta-Analysis"
        elif is_cell_line_only:
            study_type = "In Vitro / Cell Line Study"
        elif is_animal_only:
            study_type = "In Vivo / Animal Study"
        else:
            study_type = "Original Research (Standard)"

        # 2. PICO Fast-Pass Extraction
        pico_summary = {
            "Population": self._extract_population(abstract, title),
            "Intervention": self._extract_intervention(title, keywords),
            "Comparison": "Standard of Care / Control",
            "Outcome": "Clinical / Efficacy endpoints"
        }

        # 3. Soft Domain Scores Calculation (0.0 - 1.0)
        domain_scores = {}
        for domain, kw_list in self.domain_keywords.items():
            matches = sum(1 for kw in kw_list if kw in full_text)
            # Normalize match count to soft float score
            domain_scores[domain] = min(1.0, round(matches / 3.0, 2))

        return {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "pico_summary": f"P: {pico_summary['Population']}; I: {pico_summary['Intervention']}; C: {pico_summary['Comparison']}; O: {pico_summary['Outcome']}",
            "study_type": study_type,
            "is_case_report": is_case_report,
            "is_cell_line_only": is_cell_line_only,
            "is_animal_only": is_animal_only,
            "domain_scores": domain_scores
        }

    def _extract_population(self, abstract, title):
        match = re.search(r'(patients with|subjects with|individuals with|among)\s+([^.,;]+)', abstract, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:60]
        return "Target medical population"

    def _extract_intervention(self, title, keywords):
        if keywords:
            return keywords.split(';')[0].strip()
        return title[:50]

if __name__ == "__main__":
    parser = Stage0Parser()
    sample = parser.parse_paper(
        title="Genetics of migraine: where are we now?",
        abstract="Migraine is a complex brain disorder explained by genetic interaction...",
        keywords="Migraine; Genetics; Polygenic"
    )
    print("Parsed Sample Study Type:", sample["study_type"])
    print("Domain Scores:", sample["domain_scores"])
