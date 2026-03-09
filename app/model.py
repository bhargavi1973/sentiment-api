# hugging face sentiment analysis model for twitter data, fined-tuned on the latest data, with 3 classes: positive, neutral, negative
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print(f"Loading model...")

sentiment_pipeline = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    top_k=None,
    truncation=True,
    max_length=512,
)
print("Model loaded ✓")

def predict(text: str) -> dict:
    raw = sentiment_pipeline(text)[0]
    top = max(raw, key=lambda x: x["score"])
    scores = {item["label"].upper(): round(item["score"], 4) for item in raw}
    return {
        "text": text,
        "label": top["label"].upper(),
        "confidence": round(top["score"], 4),
        "scores": scores,
    }