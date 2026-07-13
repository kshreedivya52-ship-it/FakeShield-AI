import re
import string
import spacy
from bs4 import BeautifulSoup

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class TextCleaner:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        # 3. Remove HTML tags
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception:
            text = re.sub(r"<[^>]*>", "", text)
        
        # 4. Remove Numbers
        text = re.sub(r"\d+", "", text)
        
        # 5. Tokenize, remove punctuation/stopwords, and lemmatize with spaCy
        doc = nlp(text)
        
        cleaned_tokens = []
        for token in doc:
            # Remove punctuation, stopwords, whitespace
            if not token.is_stop and not token.is_punct and token.text.strip() and not token.is_space:
                cleaned_tokens.append(token.lemma_)
                
        return " ".join(cleaned_tokens)

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Hello World! Check out https://google.com for 100 new opportunities. <html><body>This is a test.</body></html>"
    print("Original:", sample)
    print("Cleaned:", cleaner.clean_text(sample))
