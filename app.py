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
origins = ["http://localhost:3000", "https://pulsevest.vercel.app", "http://10.114.6.123:3000"] 
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
async def analyze_media(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
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
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_mime_type}. Please upload audio or video.")

    except Exception as e:
        print(f"A CRITICAL ERROR OCCURRED in main handler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def analyze_audio(filename: str, mime_type: str):
    """Handles the direct Gemini analysis for audio files."""
    print("--- Running PURE Gemini Audio Analysis Pipeline ---")
    
    print(f"Uploading audio file to Google: {filename}")
    audio_file = genai.upload_file(path=filename, mime_type=mime_type)
    while audio_file.state.name == "PROCESSING":
        print("Waiting for audio processing...")
        time.sleep(5)
        audio_file = genai.get_file(audio_file.name)
    if audio_file.state.name == "FAILED":
        raise ValueError("Google Cloud file processing failed for audio.")
        
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are the senior A&R executive and music analyst at PulseVest, a leading investment firm specializing in the modern African music market. Your analysis must be objective, critical, and grounded in current industry standards. High scores should be reserved for tracks that demonstrate exceptional quality and potential; do not inflate scores.

You will be provided with an audio file for direct, in-depth review. Based ONLY on the audio content, your primary objective is to produce a detailed, professional A&R assessment formatted as a single, valid JSON object.

Do not include any introductory text, concluding remarks, or markdown formatting (like ```json) outside of the JSON structure itself.

Your analysis must cover these four core categories:

Rhythm & Groove Quality: Critically assess the rhythmic foundation of the track. Evaluate the drum programming, the effectiveness of the bassline, and the overall "pocket" or groove. Is the rhythm compelling, innovative, and well-executed? Does it possess a catchy, memorable quality that fits the intended commercial landscape?

Sound & Production Quality: Scrutinize the technical execution and production value. Is the mix clean, balanced, and dynamic, or is it cluttered and muddy? Are the vocals and instruments clear and well-processed? Does the track meet the sonic standards of a professional, radio-ready release, or does it sound like a raw demo?

Lyrical Content & Vocal Delivery: Analyze the substance and execution of the lyrics and vocals. Evaluate the theme, storytelling, wordplay, and emotional resonance of the lyrics. Assess the artist's vocal performance, including their diction, flow, cadence, and tonal quality. Do the lyrics and delivery effectively convey the song's message and emotion?

Market Potential: Forecast the track's commercial viability in the current Afrobeats/African music market. Does it have clear potential for radio airplay, streaming playlist inclusion, or viral success on platforms like TikTok and Reels? Define its likely target audience and its overall commercial appeal.

For each of the four categories, you must provide:

A score (an integer from 0 to 100).

A rationale (a concise, one- to two-sentence explanation justifying the score).

After assessing the four categories, calculate the final pulse_score by averaging the four individual category scores. The result should be a floating-point number, rounded to one decimal place.

Finally, provide a section for actionable_suggestions. This must be an object containing specific, constructive feedback for the artist. This feedback should be broken down into at least three of the following sub-keys: production_mix, arrangement, vocal_performance, lyrics, or marketing. Each suggestion must be a clear, actionable sentence designed to improve the final product.
    Your final output MUST be a single, valid JSON object with no extra text or markdown formatting. The top-level key of this object should be "analysis".
    """

    print("Contacting Gemini for audio analysis...")
    response = model.generate_content([prompt, audio_file])
    genai.delete_file(audio_file.name)
    print("Gemini analysis complete and file deleted.")
    
    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    return translate_response_for_frontend(gemini_result)


def analyze_video(filename: str, mime_type: str):
    """Handles the direct Gemini analysis for video files."""
    print("--- Running PURE Gemini Video Analysis Pipeline ---")
    
    print(f"Uploading video file to Google: {filename}")
    video_file = genai.upload_file(path=filename, mime_type=mime_type)
    while video_file.state.name == "PROCESSING":
        print("Waiting for video processing...")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError("Google Cloud file processing failed for video.")
        
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
    genai.delete_file(video_file.name)
    print("Gemini analysis complete and file deleted.")

    gemini_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    return translate_response_for_frontend(gemini_result)


def find_value_case_insensitive(data_dict: dict, target_key: str):
    """A robust function to find a key in a dictionary, ignoring case, spaces, and underscores."""
    target_key_normalized = target_key.lower().replace("_", "").replace(" ", "")
    for key, value in data_dict.items():
        key_normalized = str(key).lower().replace("_", "").replace(" ", "")
        if key_normalized == target_key_normalized:
            return value
    return None # Return None if no match is found


def translate_response_for_frontend(gemini_result: dict):
    """Takes the raw Gemini JSON and formats it perfectly for the frontend, handling inconsistencies."""
    print("--- STAGE 3: RUNNING UNBREAKABLE TRANSLATOR V2 ---")
    
    analysis_data = gemini_result.get("analysis")
    if not analysis_data:
        raise ValueError("The 'analysis' key was not found in the AI's response.")

    scores_for_frontend = []
    # We now iterate through the original data to preserve the category names
    for key, value in analysis_data.items():
        if isinstance(value, dict) and 'score' in value and 'explanation' in value:
            scores_for_frontend.append({
                "category": key,
                "score": value["score"],
                "explanation": value["explanation"]
            })

    # --- THE DEFINITIVE FIX: USING THE INTELLIGENT FINDER FUNCTION ---
    pulse_score = find_value_case_insensitive(analysis_data, "Pulse Score")
    suggestions = find_value_case_insensitive(analysis_data, "Suggestions")

    final_response = {
        "pulseScore": pulse_score,
        "suggestions": suggestions,
        "scores": scores_for_frontend
    }
    
    print(f"Translation complete. Final data: {final_response}")
    return final_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

