import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI # The correct, official client
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

# --- THE FINAL, DEFINITIVE, AND WORKING HUGGING FACE CONFIGURATION ---
# Using the OpenAI client pointed at the HF Router, as you commanded
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# --- THE ROOT ENDPOINT ---
@app.get("/")
def read_root():
    return {"status": "PulseVest Analysis Engine (Llama 3 OpenAI SDK Edition) is running"}

# --- THE ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_audio(audioFile: UploadFile = File(...)):
    temp_filename = f"temp_{audioFile.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audioFile.read())

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
        key, scale, strength = key_extractor(audio)

        essentia_data = {
            "bpm": f"{bpm:.1f}",
            "danceability": f"{danceability_result[0]:.2f}",
            "key": f"{key} {scale}",
        }
        print(f"Essentia Analysis Complete: {essentia_data}")

        # --- STAGE 2: HUGGING FACE ANALYSIS (YOUR CORRECT LOGIC) ---
        print("Contacting Llama 3 via HF Router for expert analysis...")
        
        prompt = f"""
        You are an expert A&R and music analyst for PulseVest. Your task is to analyze technical data from an audio track and return a single, valid JSON object. Do not include any text, notes, or markdown formatting before or after the JSON object. Your entire response must be only the JSON object itself.

        Here is the technical data extracted using the Essentia library: {json.dumps(essentia_data)}.

        Based ONLY on this technical data, provide a detailed assessment covering these four categories:
        1.  **Rhythm Quality:** Based on the BPM, infer the energy and potential catchiness.
        2.  **Sound Quality:** Infer this based on the context of a demo. Acknowledge this is an inference.
        3.  **Market Potential:** Based on the danceability and key, how well could this track perform in the current Afrobeats/African music market?
        4.  **Genre Relevance:** Based on all the data, what is the likely genre of this track and how does it fit?
        
        For each category, provide a score from 0 to 100 and a concise, one-sentence explanation. Calculate the final "Pulse Score" by averaging the four scores. Finally, provide a paragraph of actionable "Suggestions" for the artist.
        """
        
        completion = client.chat.completions.create(
            # --- THIS IS THE DEFINITIVE FIX: USING THE CORRECT, SUPPORTED MODEL NAME ---
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        print("Hugging Face Analysis Complete.")
        
        message_content = completion.choices[0].message.content
        
        if not message_content:
             raise ValueError("Received an empty response from the AI model.")

        # Clean the response to ensure it's valid JSON
        cleaned_json = message_content.replace('```json', '').replace('```', '').strip()
        
        return json.loads(cleaned_json)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

