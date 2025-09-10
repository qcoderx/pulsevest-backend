import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import essentia.standard as es

# --- SETUP ---
load_dotenv() # Load environment variables from .env file

app = Flask(__name__)
# Allow requests from your Next.js frontend
CORS(app, resources={r"/analyze": {"origins": "http://localhost:3000"}})

# Configure the Gemini API client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# --- THE ANALYSIS ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Receives an audio file, analyzes it with Essentia,
    gets a score from Gemini, and returns the full analysis.
    """
    if 'audioFile' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audioFile']
    
    # Create a temporary file to store the upload
    temp_filename = "temp_audio_file"
    file.save(temp_filename)

    try:
        # --- STAGE 1: ESSENTIA.JS ANALYSIS (STABLE & RELIABLE) ---
        print("Running Essentia analysis...")
        loader = es.MonoLoader(filename=temp_filename)
        audio = loader()
        
        # Extract features
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

        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_json = response.text.replace('```json', '').replace('```', '').strip()
        
        print("Gemini Analysis Complete.")
        analysis_result = json.loads(cleaned_json)

        return jsonify(analysis_result)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- RUN THE SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
