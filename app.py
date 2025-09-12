import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import essentia.standard as es
import uvicorn

# --- SETUP ---
load_dotenv()

app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000", "https://*.vercel.app"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE FINAL, DEFINITIVE, AND WORKING GEMINI CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it to your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- THE ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Analysis Engine (Gemini 1.5 Flash Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        # --- STAGE 1: ESSENTIA ANALYSIS (STABLE & RELIABLE) ---
        print("--- STAGE 1: ESSENTIA ANALYSIS ---")
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

        # --- STAGE 2: GEMINI ANALYSIS (THE FINAL ENGINE) ---
        print("\n--- STAGE 2: GEMINI ANALYSIS ---")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        
        print("Contacting Gemini 1.5 Flash...")
        response = model.generate_content(prompt)
        
        print("\n--- BLACK BOX RECORDER ---")
        print("RAW, UNFILTERED RESPONSE FROM GEMINI:")
        print(response.text)
        print("--- END OF RAW RESPONSE ---\n")
        
        cleaned_json = response.text.replace('```json', '').replace('```', '').strip()
        
        print("Attempting to parse cleaned JSON...")
        gemini_result = json.loads(cleaned_json)
        print("JSON parsing successful.")

        # --- THIS IS THE FINAL, DEFINITIVE TRANSLATOR ---
        print("\n--- STAGE 3: TRANSLATING BLUEPRINT FOR FRONTEND ---")
        
        scores_for_frontend = []
        # We iterate through the AI's response to build the correct format
        for key, value in gemini_result.items():
            if isinstance(value, dict) and 'score' in value and 'explanation' in value:
                scores_for_frontend.append({
                    "category": key,
                    "score": value["score"],
                    "explanation": value["explanation"]
                })

        # We construct the final, perfect object for the frontend
        final_response_for_frontend = {
            "pulseScore": gemini_result.get("Pulse Score"),
            "suggestions": gemini_result.get("Suggestions"),
            "scores": scores_for_frontend
        }
        
        print("Translation complete. Sending perfect data to frontend.")
        return final_response_for_frontend

    except Exception as e:
        print(f"A CRITICAL ERROR OCCURRED: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

