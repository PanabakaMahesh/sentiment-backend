import requests
import re
import os
import time

# Using cardiffnlp/twitter-roberta-base-sentiment-latest
# This model has 3 classes: positive, neutral, negative
# Much more accurate than distilbert-sst-2 which only has 2 classes
HF_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


class SentimentAnalyzer:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        print(f"SentimentAnalyzer initialized. HF_TOKEN set: {bool(HF_TOKEN)}")

    def _clean_review(self, text: str) -> str:
        """Clean and prepare review text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        # Truncate to model limit (512 tokens ~ 400 words)
        words = text.split()
        if len(words) > 400:
            text = ' '.join(words[:400])
        return text

    def _query_hf_api(self, text: str, retries: int = 3) -> list:
        """Query HuggingFace API with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=self.headers,
                    json={"inputs": text},
                    timeout=30,
                )
                result = response.json()

                # Handle model loading (first request may need to wait)
                if isinstance(result, dict) and "error" in result:
                    error_msg = result["error"]
                    if "loading" in error_msg.lower():
                        print(f"Model loading, waiting 10s... (attempt {attempt+1})")
                        time.sleep(10)
                        continue
                    else:
                        print(f"HF API error: {error_msg}")
                        return []

                # Result is [[{label, score}, {label, score}, {label, score}]]
                if isinstance(result, list) and len(result) > 0:
                    scores = result[0] if isinstance(result[0], list) else result
                    return scores

            except Exception as e:
                print(f"HF API attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

        return []

    def _get_sentiment_from_scores(self, scores: list) -> tuple:
        """
        Extract sentiment and confidence from API scores.
        Returns (sentiment, confidence).
        Model labels: positive, neutral, negative
        """
        if not scores:
            return "neutral", 0.5

        # Build a label->score map
        label_map = {}
        for item in scores:
            label = item.get("label", "").lower()
            score = item.get("score", 0.0)
            # Normalize label variations
            if label in ["positive", "pos", "label_2"]:
                label_map["positive"] = score
            elif label in ["negative", "neg", "label_0"]:
                label_map["negative"] = score
            elif label in ["neutral", "neu", "label_1"]:
                label_map["neutral"] = score

        if not label_map:
            return "neutral", 0.5

        # Get the highest scoring label
        best_label = max(label_map, key=label_map.get)
        best_score = label_map[best_label]

        pos = label_map.get("positive", 0)
        neg = label_map.get("negative", 0)
        neu = label_map.get("neutral", 0)

        print(f"  Scores -> pos:{pos:.2f} neu:{neu:.2f} neg:{neg:.2f} => {best_label} ({best_score:.2f})")

        return best_label, round(best_score, 2)

    def _check_strong_words(self, text: str) -> str | None:
        """
        Detect clearly positive/negative phrases that the model might miss.
        Returns override sentiment or None.
        """
        text_lower = text.lower()

        # Strong positive signals
        strong_positive = [
            "excellent", "outstanding", "superb", "fantastic", "amazing",
            "absolutely love", "perfect product", "best purchase", "highly recommend",
            "5 star", "five star", "worth every", "exceeded expectations",
            "mind blowing", "blown away", "totally worth", "very happy",
            "great quality", "super product", "awesome", "brilliant",
        ]

        # Strong negative signals
        strong_negative = [
            "waste of money", "worst product", "terrible", "horrible",
            "do not buy", "don't buy", "pathetic", "useless", "garbage",
            "very disappointed", "extremely disappointed", "scam",
            "broken", "stopped working", "complete waste", "fraud",
            "return immediately", "not worth", "poor quality",
        ]

        # Emojis
        positive_emojis = ['🤩', '🔥', '❤️', '😍', '👍', '✅', '💯', '🌟', '⭐', '😊', '🥰']
        negative_emojis = ['😡', '💔', '👎', '😤', '🤬', '❌', '😞', '😠', '🤦', '😒']

        for phrase in strong_negative:
            if phrase in text_lower:
                print(f"  Strong negative phrase detected: '{phrase}'")
                return "negative"

        for phrase in strong_positive:
            if phrase in text_lower:
                print(f"  Strong positive phrase detected: '{phrase}'")
                return "positive"

        for emoji in negative_emojis:
            if emoji in text:
                return "negative"

        for emoji in positive_emojis:
            if emoji in text:
                return "positive"

        return None  # No override

    def analyze_reviews(self, reviews: list) -> list:
        results = []
        total = len(reviews)

        for i, review in enumerate(reviews):
            print(f"Analyzing review {i+1}/{total}...")
            try:
                cleaned = self._clean_review(review)

                # Query HuggingFace API
                scores = self._query_hf_api(cleaned)
                sentiment, confidence = self._get_sentiment_from_scores(scores)

                # Check for strong keyword/emoji overrides
                # Only override if model confidence is below 0.75
                if confidence < 0.75:
                    override = self._check_strong_words(review)
                    if override:
                        print(f"  Override: {sentiment} -> {override}")
                        sentiment = override
                        confidence = max(confidence, 0.80)

                results.append({
                    "text": review,
                    "sentiment": sentiment,
                    "confidence": confidence,
                })

            except Exception as e:
                print(f"Analysis error on review {i+1}: {e}")
                # Fallback: use keyword check
                override = self._check_strong_words(review)
                results.append({
                    "text": review,
                    "sentiment": override if override else "neutral",
                    "confidence": 0.60,
                })

        return results
