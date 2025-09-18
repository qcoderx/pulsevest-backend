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
origins = [
    "http://localhost:3000",
    "https://pulsevest.vercel.app",
    "http://10.114.6.123:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GEMINI CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Multimodal Analysis Engine is running"}

# --- UNIFIED ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_media(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{int(time.time())}_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

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
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def analyze_audio(filename: str, mime_type: str):
    print("--- Running Gemini Audio Analysis ---")
    
    print(f"Uploading audio file: {filename}")
    audio_file = genai.upload_file(path=filename)
    while audio_file.state.name == "PROCESSING":
        print("Waiting for audio processing...")
        time.sleep(5)
        audio_file = genai.get_file(audio_file.name)
    if audio_file.state.name == "FAILED":
        raise ValueError("Google Cloud file processing failed for audio.")
        
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    You are a senior A&R executive at PulseVest, specializing in the modern African music market. Your analysis must be objective and critical. Produce a detailed A&R assessment as a single, valid JSON object. Do not include any text or markdown formatting outside of the JSON structure.

    Your JSON output must have a single top-level key: "analysis".

    The "analysis" object must contain the following keys:

    1.  "Rhythm_Groove_Quality": An object with two keys: "score" (integer 0-100) and "explanation" (a concise rationale). Assess the rhythmic foundation, drum programming, and bassline.
    2.  "Sound_Production_Quality": An object with "score" and "explanation". Scrutinize the mix, balance, and clarity. Does it meet professional, radio-ready standards?
    3.  "Lyrical_Content_Vocal_Delivery": An object with "score" and "explanation". Analyze the lyrics, theme, storytelling, and the artist's vocal performance (diction, flow, tone).
    4.  "Market_Potential": An object with "score" and "explanation". Forecast the track's commercial viability for radio, streaming, and viral platforms like TikTok.
    5.  "pulse_score": A floating-point number, rounded to one decimal place, representing the average of the four category scores.
    6.  "actionable_suggestions": A single string containing specific, constructive feedback for the artist to improve the track.
    """

    print("Contacting Gemini for audio analysis...")
    response = model.generate_content([prompt, audio_file])
    genai.delete_file(audio_file.name)
    print("Gemini analysis complete and file deleted.")
    
    # Clean the response text before parsing
    cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
    gemini_result = json.loads(cleaned_text)
    
    return translate_response_for_frontend(gemini_result)


def analyze_video(filename: str, mime_type: str):
    print("--- Running Gemini Video Analysis ---")
    
    print(f"Uploading video file: {filename}")
    video_file = genai.upload_file(path=filename)
    while video_file.state.name == "PROCESSING":
        print("Waiting for video processing...")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError("Google Cloud file processing failed for video.")
        
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = """
    You are an expert film critic for PulseVest. Based ONLY on the video content, provide a detailed assessment in a valid JSON object. Do not include any text or markdown formatting outside of the JSON structure.

    Your JSON output must have a single top-level key: "analysis".

    The "analysis" object must contain the following keys:

    1.  "Storyline_Narrative": An object with "score" (integer 0-100) and "explanation" (a concise rationale).
    2.  "Acting_Quality": An object with "score" and "explanation".
    3.  "Market_Potential": An object with "score" and "explanation".
    4.  "Cinematography_Visuals": An object with "score" and "explanation".
    5.  "pulse_score": A floating-point number, rounded to one decimal place, representing the average of the four scores.
    6.  "suggestions": A single string of actionable feedback for the filmmaker.
    """
    
    print("Contacting Gemini for video analysis...")
    response = model.generate_content([prompt, video_file])
    genai.delete_file(video_file.name)
    print("Gemini analysis complete and file deleted.")

    cleaned_text = response.text.strip().replace('```json', '').replace('```', '')
    gemini_result = json.loads(cleaned_text)
    
    return translate_response_for_frontend(gemini_result)


def translate_response_for_frontend(gemini_result: dict):
    """
    Takes the raw Gemini JSON and robustly formats it for the frontend.
    This is the definitive fix to handle potential AI inconsistencies.
    """
    print("--- Running Robust Response Translator ---")
    
    analysis_data = gemini_result.get("analysis")
    if not analysis_data or not isinstance(analysis_data, dict):
        raise ValueError("The 'analysis' key was not found or is not a dictionary.")

    scores_for_frontend = []
    
    # Iterate through all items in the analysis data
    for key, value in analysis_data.items():
        # Find any item that looks like a category score
        if isinstance(value, dict) and 'score' in value and 'explanation' in value:
            # Format the category name for readability
            category_name = key.replace('_', ' ').title()
            scores_for_frontend.append({
                "category": category_name,
                "score": value.get("score"),
                "explanation": value.get("explanation")
            })

    # Find pulse_score and suggestions using multiple possible keys
    pulse_score = analysis_data.get("pulse_score") or analysis_data.get("pulseScore")
    suggestions = analysis_data.get("actionable_suggestions") or analysis_data.get("suggestions")

    if not pulse_score or not scores_for_frontend:
        raise ValueError(f"Could not parse essential data from AI response: {analysis_data}")

    final_response = {
        "pulseScore": pulse_score,
        "suggestions": suggestions,
        "scores": scores_for_frontend
    }
    
    print(f"Translation complete. Final data: {json.dumps(final_response, indent=2)}")
    return final_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))