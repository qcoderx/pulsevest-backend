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
    Role: You are an expert A&R executive and music producer specializing in the contemporary African music scene. Your task is to provide a detailed, data-driven analysis of an audio track.

Input: An audio file.

Output: Your entire response must be a single, valid JSON object. Do not include any text or explanations outside of the JSON structure.

JSON Structure template and Instructions:

{
  "Rhythm_Groove_Quality": {
    "score": "integer (0-100)",
    "explanation": "string"
  },
  "Sound_Production_Quality": {
    "score": "integer (0-100)",
    "explanation": "string"
  },
  "Lyrical_Content_Vocal_Delivery": {
    "score": "integer (0-100)",
    "explanation": "string"
  },
  "Market_Potential": {
    "score": "integer (0-100)",
    "explanation": "string"
  },
  "pulse_score": "float (rounded to one decimal place)",
  "actionable_suggestions": "string"
}
Detailed Instructions for Each Key:

Rhythm_Groove_Quality:

Score: Assign an integer from 0-100.

Explanation: Provide a concise rationale assessing the track's rhythmic foundation. Analyze the drum programming (e.g., the bounce, pattern complexity, sound selection), the bassline's effectiveness in locking in with the drums, and the overall "pocket" or groove. Consider its effectiveness within genres like Afrobeats, Amapiano, or Afropop.

Sound_Production_Quality:

Score: Assign an integer from 0-100.

Explanation: Scrutinize the technical aspects of the production. Analyze the mix balance, clarity, stereo imaging, and dynamic range. Does it sound polished and professional enough to compete on major streaming playlists and radio? Is it "radio-ready"?

Lyrical_Content_Vocal_Delivery:

Score: Assign an integer from 0-100.

Explanation:

If vocals are present: Analyze the lyrical theme, storytelling, and depth. Evaluate the artist's vocal performance, focusing on diction, flow, pitch accuracy, emotional conviction, and tone.

If the track is instrumental: Judge the track's "vibe" and emotional weight. Assess how well lyrics and a lead vocal could potentially sit on the instrumental. Is there space for a vocalist? Does the melody suggest a strong hook? The score should reflect its potential as a compelling backing track for an Afrobeats or Pop artist.

Market_Potential:

Score: Assign an integer from 0-100.

Explanation: Forecast the track's commercial viability specifically within the African music market. Analyze its potential to succeed in genres like Afrobeats and Afropop. Consider its appeal for radio play in key markets (e.g., Nigeria, Ghana, South Africa), its potential for inclusion in major streaming playlists (e.g., Spotify's 'African Heat'), and its viral potential on platforms like TikTok and Instagram Reels. Does it have a memorable hook or instrumental motif that could trend?

pulse_score:

Calculate the average of the four scores above (Rhythm_Groove_Quality, Sound_Production_Quality, Lyrical_Content_Vocal_Delivery, Market_Potential).

The result must be a floating-point number, rounded to one decimal place.

actionable_suggestions:

Provide a single string containing specific, constructive, and practical feedback for the artist. The suggestions should directly address the weakest points identified in the analysis to help improve the track. For example: "The bassline is powerful but slightly clashes with the kick drum's low-end; try sidechain compression or a subtle EQ cut on the bass around 80Hz. To boost market appeal, consider adding a simple, repeatable log drum melody in the chorus for better TikTok potential."
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