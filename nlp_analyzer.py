import requests
import re
import os

HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set this in Render environment variables

class SentimentAnalyzer:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    def _clean_review(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:512]

    def _apply_emoji_override(self, text: str, sentiment: str) -> str:
        positive_emojis = ['🤩', '🔥', '❤️', '😍', '👍', '✅', '💯', '🌟', '⭐']
        negative_emojis = ['😡', '💔', '👎', '😤', '🤬', '❌', '😞', '😠']
        for e in positive_emojis:
            if e in text:
                return 'positive'
        for e in negative_emojis:
            if e in text:
                return 'negative'
        return sentiment

    def _query_single(self, text: str) -> dict:
        try:
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={"inputs": text},
                timeout=30
            )
            result = response.json()
            # API returns [[{label, score}, {label, score}]]
            if isinstance(result, list) and len(result) > 0:
                scores = result[0] if isinstance(result[0], list) else result
                best = max(scores, key=lambda x: x['score'])
                return best
        except Exception as e:
            print(f"HF API error: {e}")
        return {"label": "POSITIVE", "score": 0.5}

    def analyze_reviews(self, reviews: list) -> list:
        results = []
        for review in reviews:
            try:
                cleaned = self._clean_review(review)
                prediction = self._query_single(cleaned)

                label = prediction['label'].upper()
                confidence = round(prediction['score'], 2)

                if label == 'POSITIVE':
                    sentiment = 'positive'
                elif label == 'NEGATIVE':
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                # Low confidence = neutral
                if confidence < 0.65:
                    sentiment = 'neutral'

                sentiment = self._apply_emoji_override(review, sentiment)

                results.append({
                    "text": review,
                    "sentiment": sentiment,
                    "confidence": confidence,
                })
            except Exception as e:
                print(f"Analysis error: {e}")
                results.append({
                    "text": review,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                })
        return results
