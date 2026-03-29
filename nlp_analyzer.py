from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            return_all_scores=False
        )
    
    def analyze_reviews(self, reviews):
        """Analyze reviews with human-like emotion understanding"""
        results = []
        for review in reviews:
            try:
                prediction = self.model(review, truncation=True)[0]
                
                # Correct label mapping for this specific model
                label_map = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral',
                    'LABEL_2': 'positive'
                }
                
                sentiment = label_map[prediction['label']]
                confidence = prediction['score']
                
                # Human-like adjustment for emojis and emphasis
                if '🤩' in review or '🔥' in review:
                    sentiment = 'positive'
                elif '😡' in review or '💔' in review:
                    sentiment = 'negative'
                    
                results.append({
                    "text": review,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 2)
                })
            except Exception as e:
                results.append({
                    "text": review,
                    "sentiment": "neutral",
                    "confidence": 0.0
                })
        return results
