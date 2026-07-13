# pyrefly: ignore [missing-import]
from textblob import TextBlob
from typing import Dict, Any

class SentimentAnalyzer:
    def __init__(self):
        pass

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Returns polarity, subjectivity, and a sentiment label.
        """
        if not isinstance(text, str) or not text.strip():
            return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.05:
            label = "positive"
        elif polarity < -0.05:
            label = "negative"
        else:
            label = "neutral"
            
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "label": label
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(analyzer.analyze_sentiment("This is a fantastic news!"))
    print(analyzer.analyze_sentiment("This is a horrible tragedy."))
    print(analyzer.analyze_sentiment("The water boils at 100 degrees."))
