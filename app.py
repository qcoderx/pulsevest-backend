import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import essentia.standard as es
import uvicorn

# --- SETUP ---
load_dotenv()

app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000"] # Add your frontend's production URL here later
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: CONFIGURE HUGGING FACE API CLIENT ---
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN (Hugging Face Token) not found in .env file")

# Using a powerful and reliable open-source model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# --- THE ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Analysis Engine (Hugging Face Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        # --- STAGE 1: ESSENTIA ANALYSIS (STABLE & RELIABLE) ---
        print("Running Essentia analysis...")
        loader = es.MonoLoader(filename=temp_filename)
        audio = loader()
        
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, _, _, _, _ = rhythm_extractor(audio)
        danceability_algo = es.Danceability()
        danceability_result = danceability_algo(audio)
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        essentia_data = {
            "bpm": f"{bpm:.1f}",
            "danceability": f"{danceability_result[0]:.2f}",
            "key": f"{key} {scale}",
        }
        print(f"Essentia Analysis Complete: {essentia_data}")

        # --- STAGE 2: HUGGING FACE ANALYSIS (THE NEW ENGINE) ---
        print("Contacting Hugging Face for expert analysis...")
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}"
        }

        prompt = f"""
        [INST] You are an expert A&R and music analyst for PulseVest. I have analyzed an audio track and extracted the following objective data using the Essentia library: {json.dumps(essentia_data)}.

        Based ONLY on this technical data, provide a detailed assessment.
        
        Your analysis must cover these four categories:
        1.  **Rhythm Quality:** Based on the BPM, infer the energy and potential catchiness.
        2.  **Sound Quality:** Infer this based on the context of a demo. Acknowledge this is an inference.
        3.  **Market Potential:** Based on the danceability and key, how well could this track perform in the current Afrobeats/African music market?
        4.  **Genre Relevance:** Based on all the data, what is the likely genre of this track and how does it fit?
        
        For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the artist.
        
        Your final output MUST be a single, valid JSON object with no extra text or markdown formatting. [/INST]
        """

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        print("Hugging Face Analysis Complete.")
        # The response is a list, we take the first item's generated text
        generated_text = response.json()[0]['generated_text']
        
        # The model's response includes the original prompt, so we strip it out
        json_response_string = generated_text.split("[/INST]")[-1].strip()
        
        return json.loads(json_response_string)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response body: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=f"Error from Hugging Face API: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

