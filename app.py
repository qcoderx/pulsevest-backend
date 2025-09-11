import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests  # We now use the standard 'requests' library
import essentia.standard as es
import uvicorn

# --- SETUP ---
load_dotenv()

app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEW: CONFIGURE DEEPSEEK API CLIENT ---
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# --- THE ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Analysis Engine (DeepSeek Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        # --- STAGE 1: ESSENTIA ANALYSIS (UNCHANGED AND STABLE) ---
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

        # --- STAGE 2: DEEPSEEK ANALYSIS (THE NEW ENGINE) ---
        print("Contacting DeepSeek for expert analysis...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }

        prompt = f"""
        You are an expert A&R and music analyst for PulseVest. I have analyzed an audio track and extracted the following objective data using the Essentia library: {json.dumps(essentia_data)}.

        Based ONLY on this technical data, provide a detailed assessment in a valid JSON format.
        
        Your analysis must cover these four categories:
        1.  **Rhythm Quality:** Based on the BPM, infer the energy and potential catchiness.
        2.  **Sound Quality:** Infer this based on the context of a demo. Acknowledge this is an inference.
        3.  **Market Potential:** Based on the danceability and key, how well could this track perform in the current Afrobeats/African music market?
        4.  **Genre Relevance:** Based on all the data, what is the likely genre of this track and how does it fit?
        
        For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the artist.
        
        Your final output MUST be a single, valid JSON object with no extra text or markdown formatting.
        """

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that only responds with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # This will raise an exception for HTTP error codes

        print("DeepSeek Analysis Complete.")
        analysis_result = response.json()['choices'][0]['message']['content']
        
        # The content from DeepSeek is a JSON string, so we need to parse it
        return json.loads(analysis_result)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response body: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=f"Error from DeepSeek API: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

