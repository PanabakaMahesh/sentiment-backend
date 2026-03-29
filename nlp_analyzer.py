from transformers import pipeline
import re

class SentimentAnalyzer:
    def __init__(self):
        # Use a much lighter model — works within 512MB RAM on free servers
        # distilbert is 40MB vs roberta's 500MB
        self.model = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False,
        )

    def _clean_review(self, text: str) -> str:
        """Clean review text for better analysis accuracy."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Truncate very long reviews (model limit)
        return text[:512]

    def _apply_emoji_override(self, text: str, sentiment: str) -> str:
        """Adjust sentiment based on strong emoji signals."""
        positive_emojis = ['🤩', '🔥', '❤️', '😍', '👍', '✅', '💯', '🌟', '⭐']
        negative_emojis = ['😡', '💔', '👎', '😤', '🤬', '❌', '😞', '😠']
        for e in positive_emojis:
            if e in text:
                return 'positive'
        for e in negative_emojis:
            if e in text:
                return 'negative'
        return sentiment

    def analyze_reviews(self, reviews: list) -> list:
        results = []
        for review in reviews:
            try:
                cleaned = self._clean_review(review)
                prediction = self.model(cleaned, truncation=True)[0]

                # distilbert SST-2 uses POSITIVE/NEGATIVE labels
                label = prediction['label'].upper()
                confidence = round(prediction['score'], 2)

                if label == 'POSITIVE':
                    sentiment = 'positive'
                elif label == 'NEGATIVE':
                    sentiment = 'negative'
                else:
                    # Fallback for any other label format
                    sentiment = 'neutral' if confidence < 0.65 else (
                        'positive' if label in ['POSITIVE', 'LABEL_2'] else 'negative'
                    )

                # Treat low-confidence predictions as neutral
                if confidence < 0.65:
                    sentiment = 'neutral'

                # Apply emoji override for strong signals
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
