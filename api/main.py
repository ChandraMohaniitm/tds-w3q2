from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()  # 🔥 THIS LOADS .env FILE

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_sentiment(request: CommentRequest):
    try:
        # ✅ Use OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")  # 🔥 use this
        )

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",   # 🔥 required model format for OpenRouter
            messages=[
                {
                    "role": "system",
                    "content": """You are a customer feedback analyzer.
Return response strictly in JSON format like:
{
  "sentiment": "positive/negative/neutral",
  "rating": 1-5
}"""
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            temperature=0
        )

        # Parse JSON safely
        content = response.choices[0].message.content
        parsed = json.loads(content)

        sentiment = parsed.get("sentiment", "neutral").lower()
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"

        rating = int(parsed.get("rating", 3))

        return SentimentResponse(sentiment=sentiment, rating=rating)

    except Exception as e:
        # Fallback for quota/429
        if "429" in str(e) or "quota" in str(e).lower():
            text = request.comment.lower()
            sentiment = (
                "positive"
                if any(word in text for word in ["love", "amazing", "great"])
                else "neutral"
            )
            rating = 5 if sentiment == "positive" else 3
            return SentimentResponse(sentiment=sentiment, rating=rating)

        raise HTTPException(status_code=500, detail=str(e))



