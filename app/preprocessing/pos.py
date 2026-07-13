import spacy
from collections import Counter
from typing import Dict, Tuple, List

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class POSTagger:
    def __init__(self):
        pass

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Returns a list of (token, pos_tag) pairs.
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc if not token.is_space]

    def get_pos_counts(self, text: str) -> Dict[str, int]:
        """
        Returns counts of each POS tag.
        """
        tags = self.get_pos_tags(text)
        counts = Counter([tag for _, tag in tags])
        return dict(counts)

    def get_top_words_by_pos(self, text: str, pos_type: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Returns top words for a given POS type (e.g., 'NOUN', 'VERB', 'ADJ').
        """
        tags = self.get_pos_tags(text)
        filtered_words = [word.lower() for word, tag in tags if tag == pos_type and len(word) > 1]
        return Counter(filtered_words).most_common(top_n)

if __name__ == "__main__":
    tagger = POSTagger()
    sample = "The president signed the bill in Washington."
    print("POS Tags:", tagger.get_pos_tags(sample))
    print("POS Counts:", tagger.get_pos_counts(sample))
    print("Top Nouns:", tagger.get_top_words_by_pos(sample, "NOUN"))
