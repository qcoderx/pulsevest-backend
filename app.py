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
    return {"status": "PulseVest Analysis Engine (Final Advanced Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        # --- STAGE 1: DEFINITIVE, CORRECTED A LA CARTE ESSENTIA ANALYSIS ---
        print("--- STAGE 1: DEFINITIVE ADVANCED ESSENTIA ANALYSIS ---")
        
        # CRITICAL FIX: Load as STEREO audio for algorithms that require it
        loader = es.AudioLoader(filename=temp_filename)
        audio, sample_rate, _, _, _, _ = loader()
        
        # Convert stereo to mono for algorithms that need mono
        if audio.shape[0] == 2:  # If stereo
            mono_audio = es.StereoToMono()(audio)
        else:
            mono_audio = audio.mean(axis=0) if len(audio.shape) > 1 else audio
        
        print("Extracting advanced features a la carte to avoid errors...")
        
        # --- RHYTHM & KEY (using mono) ---
        rhythm_extractor = es.RhythmExtractor2013()
        bpm, _, _, _, _ = rhythm_extractor(mono_audio)
        
        danceability_algo = es.Danceability()
        danceability_result, _ = danceability_algo(mono_audio)
        
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(mono_audio)

        # --- ADVANCED SOUND QUALITY METRICS (using mono) ---
        dynamic_complexity_algo = es.DynamicComplexity()
        dynamic_complexity, _ = dynamic_complexity_algo(mono_audio)
        
        # FIX: LoudnessEBUR128 expects stereo input
        # If we have mono, duplicate it to create pseudo-stereo
        if len(audio.shape) == 1 or audio.shape[0] == 1:
            stereo_audio = np.array([mono_audio, mono_audio])
        else:
            stereo_audio = audio
        
        loudness_algo = es.LoudnessEBUR128()
        _, loudness_range, _, _ = loudness_algo(stereo_audio)

        # --- SPECTRAL ANALYSIS (using mono) ---
        # Using SpectralCentroidTime instead of SpectralContrast to avoid issues
        spectral_centroid = es.SpectralCentroidTime()(mono_audio)
        
        # Alternative: Calculate spectral complexity
        spectrum = es.Spectrum()(mono_audio)
        spectral_complexity = es.SpectralComplexity()(spectrum)

        essentia_data = {
            "bpm": f"{bpm:.1f}",
            "danceability": f"{danceability_result:.2f}",
            "key": f"{key} {scale}",
            "dynamic_complexity": f"{dynamic_complexity:.2f}",
            "loudness_range_db": f"{loudness_range:.2f}",
            "spectral_complexity": f"{spectral_complexity:.2f}",
            "spectral_centroid": f"{spectral_centroid:.2f}"
        }
        print(f"Essentia Analysis Complete: {essentia_data}")

        # --- STAGE 2: UPGRADED GEMINI ANALYSIS ---
        print("\n--- STAGE 2: UPGRADED GEMINI ANALYSIS ---")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert A&R and music analyst for PulseVest. I have analyzed an audio track and extracted the following rich, objective data using the Essentia library: {json.dumps(essentia_data)}.

        Based ONLY on this rich technical data, provide a detailed assessment in a valid JSON format.
        
        Your analysis must cover these four categories:
        1.  **Rhythm Quality:** Based on the BPM ({essentia_data['bpm']}) and Danceability score ({essentia_data['danceability']}), infer the energy and potential catchiness.
        2.  **Sound Quality:** Based on Dynamic Complexity ({essentia_data['dynamic_complexity']}) and Loudness Range ({essentia_data['loudness_range_db']} dB), assess the production quality. A higher dynamic complexity and a balanced loudness range suggest a professional mix.
        3.  **Market Potential:** Based on the danceability and key ({essentia_data['key']}), how well could this track perform in the current Afrobeats/African music market?
        4.  **Genre Relevance:** Based on all the data, especially the Spectral Complexity ({essentia_data['spectral_complexity']}) and Spectral Centroid ({essentia_data['spectral_centroid']}), use your expertise to infer the track's most likely genre and assess its sonic texture and fit. Higher values often indicate a richer, more defined sound.
        
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