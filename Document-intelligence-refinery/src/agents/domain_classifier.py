"""Domain classification utilities."""

from src.models.common import DomainHint


class DomainClassifier:
    """Simple keyword-based domain classifier."""

    def __init__(self):
        self.keywords = {
            DomainHint.FINANCIAL: [
                "revenue", "profit", "loss", "financial", "quarterly", "annual report",
                "balance sheet", "income statement", "cash flow", "earnings", "dividend",
                "stock", "share", "equity", "debt", "asset", "liability", "audit"
            ],
            DomainHint.LEGAL: [
                "legal", "law", "contract", "agreement", "clause", "litigation",
                "court", "judge", "attorney", "plaintiff", "defendant", "settlement",
                "compliance", "regulation", "statute", "ordinance"
            ],
            DomainHint.TECHNICAL: [
                "technical", "specification", "architecture", "implementation",
                "algorithm", "system", "software", "hardware", "protocol", "api",
                "framework", "methodology", "design", "development"
            ],
            DomainHint.MEDICAL: [
                "medical", "patient", "diagnosis", "treatment", "therapy", "clinical",
                "hospital", "physician", "medication", "symptom", "disease", "health"
            ],
            DomainHint.ACADEMIC: [
                "research", "study", "analysis", "methodology", "hypothesis", "conclusion",
                "citation", "reference", "bibliography", "abstract", "thesis", "dissertation"
            ],
        }

    def classify(self, text: str) -> DomainHint:
        """Classify domain based on keyword matching."""
        text_lower = text.lower()
        scores = {hint: 0 for hint in DomainHint}
        
        for hint, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[hint] += 1
        
        # Remove GENERAL from scoring
        scores.pop(DomainHint.GENERAL, None)
        
        if not scores or max(scores.values()) == 0:
            return DomainHint.GENERAL
        
        return max(scores, key=scores.get)


def create_domain_classifier(classifier_type: str = "keyword") -> DomainClassifier:
    """Create a domain classifier instance."""
    if classifier_type == "keyword":
        return DomainClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
