from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from typing import Dict

app = FastAPI(
    title="API Классификации Жалоб Граждан",
    description="Предсказание категории обращения по тексту",
    version="1.0"
)

# мод и веторайзер
model = joblib.load("category_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class PetitionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PetitionRequest) -> Dict:
    vector = vectorizer.transform([request.text])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = float(max(proba))
    
    return {
        "category": pred,
        "confidence": round(confidence, 4),
        "probabilities": {cat: round(p, 4) for cat, p in zip(model.classes_, proba)}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)