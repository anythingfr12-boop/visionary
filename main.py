import os
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image

app = FastAPI(title="Top-Tier Poketwo Naming API")

# On server startup, this loads the same deep learning pipeline used by top-tier bots
print("Loading neural network model into cloud memory...")
classifier = pipeline("image-classification", model="imzynoxprince/pokemons-image-classifier-gen1-gen9")
print("Model ready!")

class ImageRequest(BaseModel):
    url: str

@app.post("/predict")
async def predict_pokemon(request: ImageRequest):
    try:
        # Download the live, distorted image asset from Poketwo's active channel proxy
        response = requests.get(request.url, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Neural network extracts structural features, bypassing pixel-shifting anti-cheat
        predictions = classifier(img)
        
        # Extract the number one highest probability match
        top_match = predictions[0]
        pokemon_name = top_match['label'].capitalize()
        confidence = top_match['score'] * 100
        
        return {
            "pokemon": pokemon_name,
            "confidence": f"{confidence:.2f}%",
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Automatically read the port strings for free-tier compatibility
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

