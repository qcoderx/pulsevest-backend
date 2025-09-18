import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import uvicorn
import time

# --- SETUP ---
load_dotenv()

app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000", "https://pulsevest.vercel.app"] 
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# --- THE ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Production Engine (Upgraded JSON) is running"}

# --- THE UNIFIED ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_media(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer: buffer.write(await audioFile.read())

    try:
        file_mime_type = audioFile.content_type
        print(f"Received file: {audioFile.filename}, MIME Type: {file_mime_type}")

        if file_mime_type.startswith('audio/'):
            return analyze_audio(temp_filename, file_mime_type)
        elif file_mime_type.startswith('video/'):
            return analyze_video(temp_filename, file_mime_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_mime_type}.")

    except Exception as e:
        print(f"A CRITICAL ERROR OCCURRED: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

def analyze_audio(filename: str, mime_type: str):
    """Handles the direct Gemini analysis for audio files using your superior JSON structure."""
    print("--- Running PURE Gemini Audio Analysis Pipeline ---")
    
    audio_file = genai.upload_file(path=filename, mime_type=mime_type)
    while audio_file.state.name == "PROCESSING":
        print("Waiting for audio processing...")
        time.sleep(5)
        audio_file = genai.get_file(audio_file.name)
    if audio_file.state.name == "FAILED": raise ValueError("Google Cloud file processing failed for audio.")
        
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = """
    You are an expert A&R and music analyst for PulseVest. I have uploaded an audio file for your direct review. Based ONLY on listening to the audio content, provide a detailed assessment in the following valid JSON format. Do not include any text or markdown formatting before or after the JSON object.

    {
      "Rhythm_Groove_Quality": {
        "score": "integer (0-100, based on how compelling, unique, and well-executed the rhythm is)",
        "explanation": "string (a concise, one-sentence explanation for your rating)"
      },
      "Sound_Production_Quality": {
        "score": "integer (0-100, assessing the production value, mix clarity, and professional sound)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "Lyrical_Content_Vocal_Delivery": {
        "score": "integer (0-100, based on lyrical depth, vocal performance, and emotional impact)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "Market_Potential": {
        "score": "integer (0-100, assessing how well this could perform in the current Afrobeats market)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "pulse_score": "float (the calculated average of the four scores above, rounded to one decimal place)",
      "actionable_suggestions": "string (a paragraph of actionable feedback for the artist to improve the track)"
    }
    """

    print("Contacting Gemini for audio analysis...")
    response = model.generate_content([prompt, audio_file])
    genai.delete_file(audio_file.name)
    print("Gemini analysis complete and file deleted.")
    
    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    return translate_response_for_frontend(gemini_result)

def analyze_video(filename: str, mime_type: str):
    """Handles the direct Gemini analysis for video files using a similar superior JSON structure."""
    print("--- Running PURE Gemini Video Analysis Pipeline ---")
    
    video_file = genai.upload_file(path=filename, mime_type=mime_type)
    while video_file.state.name == "PROCESSING":
        print("Waiting for video processing...")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED": raise ValueError("Google Cloud file processing failed for video.")
        
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = """
    You are an expert film critic and market analyst for PulseVest. I have uploaded a video file for your review. Based ONLY on the video content, provide a detailed assessment in the following valid JSON format. Do not include any text or markdown formatting before or after the JSON object.

    {
      "Storyline_Narrative_Quality": {
        "score": "integer (0-100, based on plot, pacing, and coherence)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "Acting_Performance_Quality": {
        "score": "integer (0-100, based on believability and engagement of performances)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "Cinematography_Visuals_Quality": {
        "score": "integer (0-100, based on camera work, lighting, and overall aesthetic)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "Market_Potential": {
        "score": "integer (0-100, assessing how well this could perform in the current Nollywood/African film market)",
        "explanation": "string (a concise, one-sentence explanation)"
      },
      "pulse_score": "float (the calculated average of the four scores above, rounded to one decimal place)",
      "actionable_suggestions": "string (a paragraph of actionable feedback for the filmmaker)"
    }
    """
    
    print("Contacting Gemini for video analysis...")
    response = model.generate_content([prompt, video_file])
    genai.delete_file(video_file.name)
    print("Gemini analysis complete and file deleted.")

    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    return translate_response_for_frontend(gemini_result)

def translate_response_for_frontend(gemini_result: dict):
    """The new Master Linguist. Translates your superior JSON structure into the format the frontend needs."""
    print("--- STAGE 3: RUNNING THE UPGRADED UNBREAKABLE TRANSLATOR ---")
    
    scores_for_frontend = []
    # We iterate through the AI's response to find all the category objects
    for key, value in gemini_result.items():
        if isinstance(value, dict) and 'score' in value and 'explanation' in value:
            # We reformat the key to be more readable for the UI
            category_name = key.replace("_", " ").title()
            scores_for_frontend.append({
                "category": category_name,
                "score": value["score"],
                "explanation": value["explanation"]
            })

    # We build the final, perfect object for the frontend
    final_response = {
        "pulseScore": gemini_result.get("pulse_score"),
        "suggestions": gemini_result.get("actionable_suggestions"),
        "scores": scores_for_frontend
    }
    
    print(f"Translation complete. Final data: {final_response}")
    return final_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

