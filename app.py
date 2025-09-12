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
import time

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
    return {"status": "PulseVest Multimodal Analysis Engine is running"}

# --- THE UNIFIED ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_media(audioFile: UploadFile = File(...)): # Renamed for clarity
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

    try:
        file_mime_type = audioFile.content_type
        print(f"Received file: {audioFile.filename}, MIME Type: {file_mime_type}")

        # --- LOGIC GATE: CHOOSE THE CORRECT ASSEMBLY LINE ---
        if file_mime_type.startswith('audio/'):
            return analyze_audio(temp_filename)
        elif file_mime_type.startswith('video/'):
            return analyze_video(temp_filename, file_mime_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_mime_type}. Please upload audio or video.")

    except Exception as e:
        print(f"A CRITICAL ERROR OCCURRED in main handler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def analyze_audio(filename: str):
    """Handles the Essentia + Gemini analysis for audio files."""
    print("--- Running Audio Analysis Pipeline ---")
    
    # --- STAGE 1: ESSENTIA ANALYSIS ---
    loader = es.MonoLoader(filename=filename)
    audio = loader()
    
    rhythm_extractor = es.RhythmExtractor2013()
    bpm, _, _, _, _ = rhythm_extractor(audio)
    danceability_algo = es.Danceability()
    danceability_result, _ = danceability_algo(audio)
    key_extractor = es.KeyExtractor()
    key, scale, strength = key_extractor(audio)
    dynamic_complexity_algo = es.DynamicComplexity()
    dynamic_complexity, _ = dynamic_complexity_algo(audio)

    essentia_data = { "bpm": f"{bpm:.1f}", "danceability": f"{danceability_result:.2f}", "key": f"{key} {scale}", "dynamic_complexity": f"{dynamic_complexity:.2f}" }
    print(f"Essentia Analysis Complete: {essentia_data}")

    # --- STAGE 2: GEMINI TEXT ANALYSIS ---
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # --- THIS IS THE FULL, UNABRIDGED, AND CORRECT PROMPT ---
    prompt = f"""
    You are an expert A&R and music analyst for PulseVest. I have analyzed an audio track and extracted the following objective data using the Essentia library: {json.dumps(essentia_data)}.

    Based ONLY on this technical data, provide a detailed assessment in a valid JSON format.
    
    Your analysis must cover these four categories:
    1.  **Rhythm Quality:** Based on the BPM and Danceability score, infer the energy and potential catchiness.
    2.  **Sound Quality:** Based on the Dynamic Complexity, assess the production quality. A higher dynamic complexity suggests a professional mix.
    3.  **Market Potential:** Based on the danceability and key, how well could this track perform in the current Afrobeats/African music market?
    4.  **Genre Relevance:** Based on all the data, use your expertise to infer the track's most likely genre and assess how well it fits and innovates within that genre.
    
    For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the artist.
    
    Your final output MUST be a single, valid JSON object with no extra text or markdown formatting. The top-level key of this object should be "analysis".
    """
    
    print("Contacting Gemini for audio data analysis...")
    response = model.generate_content(prompt)
    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())

    # --- STAGE 3: TRANSLATION ---
    return translate_response_for_frontend(gemini_result)

def analyze_video(filename: str, mime_type: str):
    """Handles the direct Gemini analysis for video files."""
    print("--- Running Video Analysis Pipeline ---")

    # --- STAGE 1: UPLOAD FILE TO GOOGLE FOR ANALYSIS ---
    print(f"Uploading video file to Google: {filename}")
    video_file = genai.upload_file(path=filename, mime_type=mime_type)
    print("Video file uploaded successfully. URI:", video_file.uri)

    # We need to wait for the file to be processed before using it.
    while video_file.state.name == "PROCESSING":
        print("Waiting for video processing...")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Google Cloud file processing failed.")

    # --- STAGE 2: GEMINI VIDEO ANALYSIS ---
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    You are an expert film critic and market analyst for PulseVest. I have uploaded a video file for your review.

    Based ONLY on the video content, provide a detailed assessment in a valid JSON format.
    
    Your analysis must cover these four categories:
    1.  **Storyline & Narrative:** How compelling is the plot? Is the pacing effective? Does the story make sense?
    2.  **Acting Quality:** Assess the performances of the main actors. Are they believable and engaging?
    3.  **Market Potential:** How well could this film perform in the current Nollywood/African film market? Does it have viral or festival potential?
    4.  **Cinematography & Visuals:** How strong is the visual storytelling? Assess the camera work, lighting, color grading, and overall aesthetic.
    
    For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the filmmaker.
    
    Your final output MUST be a single, valid JSON object with no extra text or markdown formatting. The top-level key of this object should be "analysis".
    """
    
    print("Contacting Gemini for video analysis...")
    response = model.generate_content([prompt, video_file])
    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    
    # --- STAGE 3: CLEANUP AND TRANSLATION ---
    print("Deleting uploaded video file from Google Cloud...")
    genai.delete_file(video_file.name)
    print("File deleted.")
    
    return translate_response_for_frontend(gemini_result)

def translate_response_for_frontend(gemini_result: dict):
    """Takes the raw Gemini JSON and formats it perfectly for the frontend."""
    print("Translating AI response for frontend...")
    analysis_data = gemini_result.get("analysis")
    if not analysis_data:
        raise ValueError("The 'analysis' key was not found in the AI's response.")

    scores = []
    for key, value in analysis_data.items():
        if isinstance(value, dict) and 'score' in value and 'explanation' in value:
            scores.append({
                "category": key,
                "score": value["score"],
                "explanation": value["explanation"]
            })

    final_response = {
        "pulseScore": analysis_data.get("Pulse Score"),
        "suggestions": analysis_data.get("Suggestions"),
        "scores": scores
    }
    print("Translation complete.")
    return final_response

# This part is for local development, Render will use the command in render.yaml
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

