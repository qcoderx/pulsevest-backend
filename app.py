import os
import json
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import essentia.standard as es

# --- SETUP ---
load_dotenv()

app = FastAPI(title="PulseVest Analysis Engine")

# Configure CORS to allow requests from your Next.js frontend
origins = [
    "http://localhost:3000",
    # Add your deployed frontend URL here in the future
    # "https://www.pulsevest.app", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Gemini API client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    """
    Receives an audio file, saves it temporarily, analyzes it with Essentia,
    gets a score from Gemini, and returns the full analysis.
    """
    temp_filename = f"temp_{audioFile.filename}"
    
    # Save the uploaded file temporarily
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audioFile.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")

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
        key, scale = key_extractor(audio)

        essentia_data = {
            "bpm": f"{bpm:.1f}",
            "danceability": f"{danceability_result[0]:.2f}",
            "key": f"{key} {scale}",
        }
        print(f"Essentia Analysis Complete: {essentia_data}")

        # --- STAGE 2: GEMINI ANALYSIS (INTELLIGENT SCORING) ---
        print("Contacting Gemini for expert analysis...")
        model = genai.GenerativeModel('gemini-pro')
        
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

        response = await model.generate_content_async(prompt)
        cleaned_json = response.text.replace('```json', '').replace('```', '').strip()
        
        print("Gemini Analysis Complete.")
        analysis_result = json.loads(cleaned_json)

        return analysis_result

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- A simple health check endpoint ---
@app.get("/")
def read_root():
    return {"message": "PulseVest FastAPI Analysis Engine is running."}

