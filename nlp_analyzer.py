import requests
import re
import os
import json

# ---------------------------------------------------------------------------
# We use the Anthropic Claude API for accurate sentiment analysis.
# Claude understands full sentence context, sarcasm, mixed emotions, etc.
# Get a free API key at console.anthropic.com
# Set it as ANTHROPIC_API_KEY in Render environment variables.
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

STRONG_POSITIVE_WORDS = [
    "superb", "super", "excellent", "outstanding", "amazing", "fantastic",
    "brilliant", "perfect", "awesome", "wonderful", "love it", "best",
    "impressive", "flawless", "exceptional", "highly recommend", "great product",
    "value for money", "worth every", "fast delivery", "works perfectly",
    "very happy", "very satisfied", "good quality", "nice product",
    "good product", "sturdy", "comfortable", "happy with", "loving it",
]

STRONG_NEGATIVE_WORDS = [
    "worst", "terrible", "horrible", "pathetic", "useless", "waste",
    "fraud", "fake", "broken", "damaged", "defective", "poor quality",
    "do not buy", "don't buy", "not worth", "disappointed", "cheated",
    "bad product", "bad quality", "stopped working", "not working",
    "returned", "refund", "scam", "waste of money", "very bad",
    "not good", "falling apart", "cheaply made",
]

POSITIVE_EMOJIS = ['🤩','🔥','❤️','😍','👍','✅','💯','🌟','⭐','😊','🥰','💪','🎉','😄','🙌']
NEGATIVE_EMOJIS = ['😡','💔','👎','😤','🤬','❌','😞','😠','🙁','😢','💀','🤮','😒']


class SentimentAnalyzer:
    def __init__(self):
        if ANTHROPIC_API_KEY:
            print("✓ Using Claude API for sentiment analysis")
        else:
            print("⚠ No ANTHROPIC_API_KEY found — using keyword analysis only")

    def _clean_review(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:600]

    def _emoji_sentiment(self, text: str):
        pos = sum(1 for e in POSITIVE_EMOJIS if e in text)
        neg = sum(1 for e in NEGATIVE_EMOJIS if e in text)
        if pos > neg and pos > 0:
            return 'positive', 0.90
        elif neg > pos and neg > 0:
            return 'negative', 0.90
        return None

    def _keyword_sentiment(self, text: str):
        t = text.lower()
        pos = sum(1 for w in STRONG_POSITIVE_WORDS if w in t)
        neg = sum(1 for w in STRONG_NEGATIVE_WORDS if w in t)
        if pos > 0 and neg == 0:
            return 'positive', min(0.75 + pos * 0.05, 0.95)
        elif neg > 0 and pos == 0:
            return 'negative', min(0.75 + neg * 0.05, 0.95)
        elif pos > neg:
            return 'positive', 0.70
        elif neg > pos:
            return 'negative', 0.70
        return None

    def _claude_sentiment_batch(self, reviews: list) -> list:
        """
        Use Claude to analyze up to 20 reviews at once for efficiency.
        Claude understands full sentence context, sarcasm, and mixed emotions.
        """
        if not ANTHROPIC_API_KEY:
            return None

        numbered = "\n".join([f"{i+1}. {r[:300]}" for i, r in enumerate(reviews)])

        prompt = f"""You are a sentiment analysis expert for Amazon product reviews.

Analyze each review below and classify it as exactly one of: positive, neutral, or negative.

Rules:
- "positive": Overall happy, satisfied, recommends the product, praises quality/value
- "negative": Unhappy, disappointed, complains about quality/delivery/damage, warns others
- "neutral": Mixed feelings, neither clearly happy nor unhappy, just stating facts
- Consider the FULL sentence meaning, not just individual words
- "good but broken headrest" = NEGATIVE (problem outweighs positive)
- "value for money" = POSITIVE
- "comfortable and easy to assemble" = POSITIVE
- "good product" = POSITIVE
- "sturdy and comfortable" = POSITIVE
- "overall happy" = POSITIVE

Return ONLY a valid JSON array with {len(reviews)} objects, one per review, in order:
[{{"sentiment": "positive", "confidence": 0.92}}, ...]

confidence should be between 0.70 and 0.99.

Reviews to analyze:
{numbered}"""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )

            data = response.json()
            if "content" not in data:
                print(f"Claude API error: {data}")
                return None

            raw = data["content"][0]["text"].strip()
            # Extract JSON array even if there's extra text
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not match:
                print(f"Could not find JSON in response: {raw[:200]}")
                return None

            results = json.loads(match.group())
            if len(results) == len(reviews):
                return results
            print(f"Result count mismatch: expected {len(reviews)}, got {len(results)}")
            return None

        except Exception as e:
            print(f"Claude API error: {e}")
            return None

    def analyze_reviews(self, reviews: list) -> list:
        results = []
        # Separate reviews that need AI vs those handled by rules
        ai_needed_indices = []
        ai_needed_reviews = []
        pre_results = {}

        for i, review in enumerate(reviews):
            cleaned = self._clean_review(review)

            # Priority 1: Emoji (most explicit signal)
            emoji = self._emoji_sentiment(review)
            if emoji:
                pre_results[i] = {"text": review, "sentiment": emoji[0], "confidence": round(emoji[1], 2)}
                continue

            # Priority 2: Strong keywords
            keyword = self._keyword_sentiment(cleaned)
            if keyword:
                pre_results[i] = {"text": review, "sentiment": keyword[0], "confidence": round(keyword[1], 2)}
                continue

            # Needs Claude analysis
            ai_needed_indices.append(i)
            ai_needed_reviews.append(cleaned)

        # Batch Claude analysis in groups of 20
        ai_results_map = {}
        if ai_needed_reviews and ANTHROPIC_API_KEY:
            batch_size = 20
            flat_results = []
            for b in range(0, len(ai_needed_reviews), batch_size):
                batch = ai_needed_reviews[b:b+batch_size]
                batch_results = self._claude_sentiment_batch(batch)
                if batch_results:
                    flat_results.extend(batch_results)
                else:
                    # Fallback: mark as neutral
                    flat_results.extend([{"sentiment": "neutral", "confidence": 0.5}] * len(batch))

            for idx, result in zip(ai_needed_indices, flat_results):
                ai_results_map[idx] = result

        # Assemble final results in original order
        for i, review in enumerate(reviews):
            if i in pre_results:
                results.append(pre_results[i])
            elif i in ai_results_map:
                r = ai_results_map[i]
                results.append({
                    "text": review,
                    "sentiment": r.get("sentiment", "neutral"),
                    "confidence": round(float(r.get("confidence", 0.75)), 2),
                })
            else:
                # Pure fallback if no AI key
                keyword = self._keyword_sentiment(self._clean_review(review))
                if keyword:
                    results.append({"text": review, "sentiment": keyword[0], "confidence": round(keyword[1], 2)})
                else:
                    results.append({"text": review, "sentiment": "neutral", "confidence": 0.5})

        return results
