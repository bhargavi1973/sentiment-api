from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {'status' : 'ok'}

@app.post("/predict")
def predict_sentiment(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(payload.text) > 512:
        raise HTTPException(status_code=400, detail="Text too long.")
    return predict(payload.text)