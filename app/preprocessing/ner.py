import spacy
from collections import Counter
from typing import List, Dict, Tuple

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
except OSError:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])

class NERExtractor:
    def __init__(self):
        pass

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns a list of (entity_text, entity_label) pairs.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_entity_counts(self, text: str) -> Dict[str, int]:
        """
        Returns counts of each entity type (e.g., 'PERSON', 'ORG').
        """
        ents = self.extract_entities(text)
        counts = Counter([label for _, label in ents])
        return dict(counts)

    def get_top_entities_by_type(self, text: str, ent_type: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Returns the top entities for a given type.
        """
        ents = self.extract_entities(text)
        filtered = [text for text, label in ents if label == ent_type]
        return Counter(filtered).most_common(top_n)

if __name__ == "__main__":
    ner = NERExtractor()
    sample = "Joe Biden visited Washington to meet with the WHO and NASA in India."
    print("Entities:", ner.extract_entities(sample))
    print("Counts:", ner.get_entity_counts(sample))
    print("Top ORGs:", ner.get_top_entities_by_type(sample, "ORG"))
