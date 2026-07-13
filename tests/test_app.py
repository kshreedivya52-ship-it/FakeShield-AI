import pytest
import pandas as pd
from app.preprocessing.clean import TextCleaner
from app.preprocessing.pos import POSTagger
from app.preprocessing.ner import NERExtractor
from app.preprocessing.sentiment import SentimentAnalyzer
from app.utils.data_loader import DataLoader


def test_data_loader():
    loader = DataLoader()
    df = loader.load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "label" in df.columns
    assert "text" in df.columns


def test_text_cleaner():
    cleaner = TextCleaner()
    text = "Hello World! Visit http://example.com for 123 opportunities."
    cleaned = cleaner.clean_text(text)
    assert "example" not in cleaned
    assert "opportunities" in cleaned or "opportunity" in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned


def test_pos_tagger():
    tagger = POSTagger()
    text = "The quick brown fox jumps over the lazy dog."
    tags = tagger.get_pos_tags(text)
    assert len(tags) > 0
    counts = tagger.get_pos_counts(text)
    assert isinstance(counts, dict)
    assert "NOUN" in counts or "PROPN" in counts


def test_ner_extractor():
    ner = NERExtractor()
    text = "Donald Trump visited Washington D.C."
    entities = ner.extract_entities(text)
    assert len(entities) > 0
    labels = [label for _, label in entities]
    assert "PERSON" in labels or "GPE" in labels or "ORG" in labels


def test_sentiment_analyzer():
    analyzer = SentimentAnalyzer()
    res1 = analyzer.analyze_sentiment("This news is absolutely fantastic and wonderful!")
    assert res1["polarity"] > 0
    assert res1["label"] == "positive"

    res2 = analyzer.analyze_sentiment("This is a horrible disaster and a complete failure.")
    assert res2["polarity"] < 0
    assert res2["label"] == "negative"
