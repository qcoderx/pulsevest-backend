import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import essentia.standard as es
import uvicorn
import numpy as np

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
    return {"status": "PulseVest Analysis Engine (Final Unbreakable Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        # --- STAGE 1: DEFINITIVE, UNBREAKABLE A LA CARTE ESSENTIA ANALYSIS ---
        print("--- STAGE 1: DEFINITIVE ADVANCED ESSENTIA ANALYSIS ---")
        
        # Using MonoLoader is correct and robust for these algorithms
        loader = es.MonoLoader(filename=temp_filename)
        audio = loader()
        
        print("Extracting advanced features a la carte to guarantee compatibility...")
        
        # --- RHYTHM & KEY (ALL MONO-COMPATIBLE) ---
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, _, _, _, _ = rhythm_extractor(audio)
        danceability_algo = es.Danceability()
        danceability_result, _ = danceability_algo(audio)
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)

        # --- ADVANCED SOUND QUALITY METRICS (ALL MONO-COMPATIBLE) ---
        dynamic_complexity_algo = es.DynamicComplexity()
        dynamic_complexity, _ = dynamic_complexity_algo(audio)
        
        spectral_contrast_algo = es.SpectralContrast()
        spec_contrast, _, _, _ = spec_contrast_algo(audio)
        avg_spectral_contrast = np.mean(spec_contrast)

        spectral_flatness_algo = es.Flatness()
        spectral_flatness = np.mean(spectral_flatness_algo(audio))

        essentia_data = {
            "bpm": f"{bpm:.1f}",
            "danceability": f"{danceability_result:.2f}",
            "key": f"{key} {scale}",
            "dynamic_complexity": f"{dynamic_complexity:.2f}",
            "spectral_contrast": f"{avg_spectral_contrast:.2f}",
            "spectral_flatness": f"{spectral_flatness:.4f}"
        }
        print(f"Essentia Analysis Complete: {essentia_data}")

        # --- STAGE 2: UPGRADED GEMINI ANALYSIS ---
        print("\n--- STAGE 2: UPGRADED GEMINI ANALYSIS ---")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert A&R and music analyst for PulseVest. I have analyzed an audio track and extracted the following rich, objective data using the Essentia library: {json.dumps(essentia_data)}.

        Based ONLY on this rich technical data, provide a detailed assessment in a valid JSON format.
        
        Your analysis must cover these four categories:
        1.  **Rhythm Quality:** Based on the BPM ({essentia_data['bpm']}) and Danceability score ({essentia_data['danceability']}), infer the energy and potential catchiness.
        2.  **Sound Quality:** Based on Dynamic Complexity ({essentia_data['dynamic_complexity']}) and Spectral Flatness ({essentia_data['spectral_flatness']}), assess the production quality. A higher dynamic complexity suggests a professional mix, while a lower spectral flatness indicates a more tonal, less noisy sound.
        3.  **Market Potential:** Based on the danceability and key ({essentia_data['key']}), how well could this track perform in the current Afrobeats/African music market?
        4.  **Genre Relevance:** Based on all the data, especially the Spectral Contrast ({essentia_data['spectral_contrast']}), use your expertise to infer the track's most likely genre and assess its sonic texture and fit.
        
        For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the artist.
        
        Your final output MUST be a single, valid JSON object with no extra text or markdown formatting. The top-level key of this object should be "analysis".
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

        # --- STAGE 3: THE FINAL, DEFINITIVE TRANSLATOR ---
        print("\n--- STAGE 3: TRANSLATING BLUEPRINT FOR FRONTEND ---")
        
        analysis_data = gemini_result.get("analysis")
        if not analysis_data:
            raise ValueError("The 'analysis' key was not found in the AI's response.")

        scores_for_frontend = []
        for key, value in analysis_data.items():
            if isinstance(value, dict) and 'score' in value and 'explanation' in value:
                scores_for_frontend.append({
                    "category": key,
                    "score": value["score"],
                    "explanation": value["explanation"]
                })

        final_response_for_frontend = {
            "pulseScore": analysis_data.get("Pulse Score"),
            "suggestions": analysis_data.get("Suggestions"),
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

