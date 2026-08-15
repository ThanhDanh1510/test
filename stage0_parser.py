import re

class Stage0Parser:
    """
    Stage 0: Structured Paper Understanding (DTAR v3.0)
    Extracts PICO, Study Type, MeSH categories, soft Domain probabilities, and soft Paper Signals.
    """
    def __init__(self):
        # 9 Major Biomedical Domains
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
        Parses raw text into structured representation for DTAR v3.0 decision pipeline.
        """
        full_text = f"{title} {abstract} {keywords}".lower()

        # 1. Soft Paper Signals (Probabilities / Confidence scores [0.0 - 1.0])
        # Case Report signal
        case_matches = len(re.findall(r'\b(case report|case series|clinical image|photo essay)\b', full_text))
        is_case_report_score = min(1.0, round(case_matches * 0.5, 2)) if case_matches > 0 else 0.01

        # Cell Line signal
        cell_matches = len(re.findall(r'\b(in vitro|cell line|cell culture|hela|hek293|primary culture)\b', full_text))
        human_matches = len(re.findall(r'\b(patient|clinical trial|human subjects|cohort)\b', full_text))
        is_cell_line_score = min(1.0, round(cell_matches * 0.4, 2)) if cell_matches > 0 and human_matches == 0 else 0.02

        # Review / Meta-analysis signal
        review_matches = len(re.findall(r'\b(systematic review|meta-analysis|narrative review|literature review)\b', full_text))
        is_review_score = min(1.0, round(review_matches * 0.5, 2)) if review_matches > 0 else 0.01

        # Clinical Trial signal
        trial_matches = len(re.findall(r'\b(randomized controlled trial|rct|double-blind|phase [123]|clinical trial)\b', full_text))
        is_trial_score = min(1.0, round(trial_matches * 0.45 + 0.1, 2)) if trial_matches > 0 else 0.05

        # Animal study signal
        animal_matches = len(re.findall(r'\b(rat|rats|mice|mouse|murine|canine|porcine|in vivo animal)\b', full_text))
        is_animal_score = min(1.0, round(animal_matches * 0.4, 2)) if animal_matches > 0 and human_matches == 0 else 0.01

        # 2. Study Type Classification
        if is_case_report_score >= 0.5:
            study_type = "Case Report"
        elif is_trial_score >= 0.5:
            study_type = "Randomized Controlled Trial"
        elif is_review_score >= 0.5:
            study_type = "Systematic Review / Meta-Analysis"
        elif is_cell_line_score >= 0.5:
            study_type = "In Vitro / Cell Line Study"
        elif is_animal_score >= 0.5:
            study_type = "In Vivo / Animal Study"
        else:
            study_type = "Original Clinical Research"

        # 3. PICO Extraction
        pop_str = self._extract_population(abstract, title)
        int_str = self._extract_intervention(title, keywords)
        pico_dict = {
            "population": pop_str,
            "intervention": int_str,
            "comparison": "Standard of Care / Placebo Control",
            "outcome": "Clinical & Efficacy Endpoints"
        }

        # 4. Soft Domain Scores Calculation (0.0 - 1.0)
        domains_dict = {}
        for domain, kw_list in self.domain_keywords.items():
            matches = sum(1 for kw in kw_list if kw in full_text)
            domains_dict[domain] = min(1.0, round(matches / 3.0, 2))

        # MeSH Category Keywords
        mesh_terms = [k.strip() for k in keywords.split(';') if k.strip()] if keywords else [w.title() for w in title.split()[:4]]

        return {
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "study_type": study_type,
            "pico": pico_dict,
            "mesh_terms": mesh_terms,
            "domains": domains_dict,
            "paper_signals": {
                "is_case_report": is_case_report_score,
                "is_cell_line": is_cell_line_score,
                "is_review": is_review_score,
                "is_clinical_trial": is_trial_score,
                "is_animal_only": is_animal_score
            }
        }

    def _extract_population(self, abstract, title):
        match = re.search(r'(patients with|subjects with|individuals with|among|cohort of)\s+([^.,;]+)', abstract, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:60]
        return "Target clinical patient cohort"

    def _extract_intervention(self, title, keywords):
        if keywords:
            return keywords.split(';')[0].strip()
        return title[:50]

if __name__ == "__main__":
    parser = Stage0Parser()
    sample = parser.parse_paper(
        title="Genetics of migraine: where are we now?",
        abstract="Migraine is a complex brain disorder explained by genetic interaction in patient cohorts...",
        keywords="Migraine; Genetics; Polygenic"
    )
    print("Parsed Sample Study Type:", sample["study_type"])
    print("Signals:", sample["paper_signals"])
