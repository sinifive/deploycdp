from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import google.generativeai as genai
from flask_cors import CORS
import numpy as np
import re
import logging
import base64 # For encoding/decoding audio

# Import Google Cloud Translation
from google.cloud import translate_v2 as translate
# Import Google Cloud Speech-to-Text
from google.cloud import speech
# Import Google Cloud Text-to-Speech
from google.cloud import texttospeech

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Google Gemini API Key
# IMPORTANT: Replace with your actual Gemini API Key.
# For production, consider using environment variables or a secure configuration management.
GEMINI_API_KEY = "AIzaSyD5x6QSdvJdAsIHrLOlrpds8snlIOYitvg"

# Initialize Google Cloud clients
# For production, it's highly recommended to use a service account by setting
# the GOOGLE_APPLICATION_CREDENTIALS environment variable.
# Example: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
translate_client = None
speech_client = None
tts_client = None

try:
    translate_client = translate.Client()
    logging.info("Google Cloud Translation client configured successfully.")
except Exception as e:
    logging.error(f"Error initializing Google Cloud Translation client: {e}")
    logging.error("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set for Translation.")

try:
    speech_client = speech.SpeechClient()
    logging.info("Google Cloud Speech-to-Text client configured successfully.")
except Exception as e:
    logging.error(f"Error initializing Google Cloud Speech-to-Text client: {e}")
    logging.error("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set for Speech-to-Text.")

try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Google Cloud Text-to-Speech client configured successfully.")
except Exception as e:
    logging.error(f"Error initializing Google Cloud Text-to-Speech client: {e}")
    logging.error("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set for Text-to-Speech.")


# Configure Gemini API
gemini_model = None
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured successfully.")

    # Choose your preferred Gemini model here
    # Using a preview model as requested, ensure it's available for your API key
    PREFERRED_GEMINI_MODEL = 'models/gemini-2.5-pro-preview-03-25' # Or 'gemini-pro' for general use
    gemini_model = genai.GenerativeModel(PREFERRED_GEMINI_MODEL)
    logging.info(f"Attempting to use Gemini model: {PREFERRED_GEMINI_MODEL}")

except Exception as e:
    logging.error(f"An error occurred during Gemini API configuration or model listing: {e}")
    logging.error("Please double-check your Gemini API key and network connection.")


# Model and class names paths
# Ensure these paths are correct relative to your app.py or are absolute paths
MODEL_PATH = "model_output/model_scripted.pth"
CLASS_NAMES_PATH = "model_output/class_names.txt"

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device for PyTorch model: {device}")

model = None
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    logging.info("PyTorch model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading PyTorch model from {MODEL_PATH}: {e}")

# Load class names
class_names = {}
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        for line in f:
            index, name = line.strip().split(':', 1)
            class_names[int(index)] = name
    logging.info(f"Loaded {len(class_names)} class names from {CLASS_NAMES_PATH}.")
except Exception as e:
    logging.error(f"Error loading class names from {CLASS_NAMES_PATH}: {e}")

# Image transformation pipeline for the PyTorch model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def translate_text(text, target_language, source_language=None):
    """
    Translates text using Google Cloud Translation API.
    Args:
        text (str): The text to translate.
        target_language (str): The BCP-47 language code for the target language (e.g., 'en', 'hi', 'te').
                                This should be the base language code (e.g., 'en' not 'en-US').
        source_language (str, optional): The BCP-47 language code for the source language.
                                          If None, the API will attempt to detect it.
    Returns:
        str: The translated text, or the original text if translation fails or client is not initialized.
    """
    if translate_client is None:
        logging.warning("Translation client not initialized. Returning original text.")
        return text

    if not text:
        return "" # Return empty string if input text is empty

    try:
        result = translate_client.translate(text, target_language=target_language, source_language=source_language)
        return result['translatedText']
    except Exception as e:
        logging.error(f"Translation API error for text '{text[:50]}...': {e}")
        return text # Return original text on error

def query_gemini(prompt):
    """
    Sends a prompt to the configured Gemini model.
    Args:
        prompt (str): The text prompt for Gemini.
    Returns:
        str: The response text from Gemini, or an error message if the API is not initialized or an error occurs.
    """
    if gemini_model is None:
        logging.error("Gemini API not initialized. Cannot query model.")
        return "Gemini API not initialized due to an error."
    try:
        response = gemini_model.generate_content(prompt)
        if response and hasattr(response, 'candidates') and response.candidates:
            try:
                return response.candidates[0].content.parts[0].text
            except (AttributeError, IndexError):
                logging.error("Unexpected Gemini API response structure.")
                return "Unexpected Gemini API response structure."
        else:
            logging.error("Gemini API returned no candidates.")
            return "No response from Gemini API."
    except Exception as e:
        logging.error(f"Gemini API error during query '{prompt[:50]}...': {e}")
        # Log the full response object for debugging if available
        if hasattr(response, 'text'):
            logging.error(f"Gemini raw response: {response.text}")
        return f"Error communicating with Gemini API: {e}"

def get_treatment_advice(disease_name, target_language_full='en-US'):
    """
    Generates brief treatment advice for a given disease name, handling translation.
    Args:
        disease_name (str): The name of the crop disease (expected in English).
        target_language_full (str): The desired full language code for the advice (e.g., 'en-US', 'hi-IN').
    Returns:
        str: The treatment advice in the target language.
    """
    # Extract base language code for translation (e.g., 'en' from 'en-US')
    base_target_lang = target_language_full.split('-')[0]

    # The disease_name is already expected to be in English from the classification model.
    disease_name_for_gemini = disease_name

    prompt = f"""Provide a brief "treatment" plan for a crop disease identified as **{disease_name_for_gemini}**.
    Format the advice clearly using headings and bullet points for easy understanding.
    Give only 4 or 5 concise points of treatment suggestions, not going in-depth, roughly 10 lines total."""
    
    response_en = query_gemini(prompt)

    # Translate the Gemini response back to the target language if needed
    final_response = response_en
    if base_target_lang != 'en':
        final_response = translate_text(response_en, base_target_lang, 'en')
        logging.info(f"Translated Gemini response for treatment advice to '{base_target_lang}'.")

    return final_response

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles crop disease prediction from an uploaded image.
    Performs image classification and generates treatment advice.
    """
    if model is None:
        logging.error("Prediction request failed: PyTorch model not loaded.")
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        logging.warning("Prediction request failed: No image provided.")
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        try:
            img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        except Exception as img_err:
            logging.error(f"Invalid image file: {img_err}")
            return jsonify({"error": "Invalid image file."}), 400

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class_index].item() * 100)

        disease_name = class_names.get(predicted_class_index, f"Unknown (Class {predicted_class_index})").replace('_', ' ')
        logging.info(f"Image classified as: {disease_name} with confidence: {confidence:.2f}%")

        target_language_full = request.form.get('target_language', 'en-US')
        base_target_lang = target_language_full.split('-')[0]

        # Translate disease name for display
        if base_target_lang != 'en':
            disease_name_display = translate_text(disease_name, base_target_lang, 'en')
        else:
            disease_name_display = disease_name

        treatment_advice = get_treatment_advice(disease_name, target_language_full)
        logging.info(f"Treatment advice generated for {disease_name}.")

        result = {
            'disease': disease_name,  # English name for internal use
            'disease_display': disease_name_display,  # Translated name for UI
            'confidence': confidence,
            'treatment_advice': treatment_advice
        }

        return jsonify(result)

    except Exception as e:
        logging.exception("Error during image prediction:") # Logs traceback
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Handles chatbot interactions, translating user queries and bot responses.
    """
    data = request.json
    user_query = data.get("query", "")
    disease_context = data.get("disease", "") # This is the original English disease name
    # Source language of the user's query (e.g., 'en-US', 'hi-IN')
    source_language_full = data.get("source_language", "en-US")
    # Base language code for translation (e.g., 'en', 'hi')
    base_source_lang = source_language_full.split('-')[0]

    if not user_query:
        logging.warning("Chatbot request failed: Empty query received.")
        return jsonify({"error": "Empty query"}), 400

    # Translate user query to English for Gemini if not already English
    user_query_for_gemini = user_query
    if base_source_lang != 'en':
        translated_query = translate_text(user_query, 'en', base_source_lang)
        if translated_query and translated_query.lower() != user_query.lower():
            user_query_for_gemini = translated_query
            logging.info(f"Translated user query to English for Gemini: '{user_query_for_gemini[:50]}...'")
        else:
            logging.warning(f"Could not translate user query '{user_query[:50]}...' to English. Using original.")

    # Disease context is already expected to be in English from the /predict endpoint
    # So, no translation needed for disease_context_for_gemini

    prompt = f"You're a crop expert. Answer this query: '{user_query_for_gemini}'"
    if disease_context: # disease_context is already in English
        prompt = f"The crop has {disease_context}. {prompt}"

    response_en = query_gemini(prompt)
    logging.info(f"Gemini responded to chatbot query: '{response_en[:50]}...'")

    # Translate Gemini's English response back to the user's source language
    final_response = response_en
    if base_source_lang != 'en':
        final_response = translate_text(response_en, base_source_lang, 'en')
        logging.info(f"Translated Gemini response for chatbot to '{base_source_lang}'.")

    return jsonify({"response": final_response})

@app.route('/disease_info', methods=['POST'])
def disease_info():
    """
    Provides comprehensive information about a crop disease, handling translation.
    This function now parses the English response first, then translates individual parts.
    """
    data = request.json
    disease_name = data.get("disease", "") # This should be the original English disease name
    # Target language for the response (e.g., 'en-US', 'hi-IN')
    target_language_full = data.get("target_language", "en-US")
    # Base language code for translation (e.g., 'en', 'hi')
    base_target_lang = target_language_full.split('-')[0]

    if not disease_name:
        logging.warning("Disease info request failed: No disease specified.")
        return jsonify({"error": "No disease specified"}), 400

    # The disease_name passed from the frontend (currentDisease) is the English name
    # from the classification, so no translation needed for the Gemini prompt.
    disease_name_for_gemini = disease_name

    prompt = f"""Provide comprehensive information about the crop disease '{disease_name_for_gemini}'.
STRICTLY use these headings (with double asterisks and colon): 
**Cause:**, **Symptoms:**, **Conditions Favoring Development:**, **Management Strategies:**, **Cultural Control:**, **Chemical Control:**, **Biological Control:**, **Long-Term Prevention:**
...
"""

    response_text_en = query_gemini(prompt)
    logging.info(f"Gemini responded to disease info query (English): '{response_text_en}'")  # Log full response

    # Parse the English response text into a structured dictionary first
    # This is crucial for robust parsing before translation
    disease_info_dict_en = {
        "cause": "",
        "symptoms": [],
        "conditions": [],
        "cultural_control": [],
        "chemical_control": [],
        "biological_control": [],
        "prevention": []
    }

    # Use regular expressions to extract information based on ENGLISH headings
    cause_match = re.search(r"[*_]*Cause[:：]*[*_]*\s*(.+?)(?=[*_]*Symptoms[:：]*[*_]*|$)", response_text_en, re.IGNORECASE | re.DOTALL)
    if cause_match:
        disease_info_dict_en["cause"] = cause_match.group(1).strip()
    else:
        disease_info_dict_en["cause"] = "Information not found."

    symptoms_match = re.search(r"[*_]*Symptoms[:：]*[*_]*\s*(.+?)(?=[*_]*Conditions Favoring Development[:：]*[*_]*|$)", response_text_en, re.IGNORECASE | re.DOTALL)
    if symptoms_match:
        disease_info_dict_en["symptoms"] = [item.strip().lstrip('- ') for item in symptoms_match.group(1).strip().split('\n') if item.strip()]

    conditions_match = re.search(r"[*_]*Conditions Favoring Development[:：]*[*_]*\s*(.+?)(?=[*_]*Management Strategies[:：]*[*_]*|$)", response_text_en, re.IGNORECASE | re.DOTALL)
    if conditions_match:
        disease_info_dict_en["conditions"] = [item.strip().lstrip('- ') for item in conditions_match.group(1).strip().split('\n') if item.strip()]

    management_match = re.search(r"[*_]*Management Strategies[:：]*[*_]*\s*(.+?)(?=[*_]*Long-Term Prevention[:：]*[*_]*|$)", response_text_en, re.IGNORECASE | re.DOTALL)
    if management_match:
        cultural_match = re.search(r"[*_]*Cultural Control[:：]*[*_]*\s*(.+?)(?=[*_]*Chemical Control[:：]*[*_]*|$)", management_match.group(1), re.IGNORECASE | re.DOTALL)
        if cultural_match:
            disease_info_dict_en["cultural_control"] = [item.strip().lstrip('- ') for item in cultural_match.group(1).strip().split('\n') if item.strip()]

        chemical_match = re.search(r"[*_]*Chemical Control[:：]*[*_]*\s*(.+?)(?=[*_]*Biological Control[:：]*[*_]*|$)", management_match.group(1), re.IGNORECASE | re.DOTALL)
        if chemical_match:
            disease_info_dict_en["chemical_control"] = [item.strip().lstrip('- ') for item in chemical_match.group(1).strip().split('\n') if item.strip()]

        biological_match = re.search(r"[*_]*Biological Control[:：]*[*_]*\s*(.+)", management_match.group(1), re.IGNORECASE | re.DOTALL)
        if biological_match:
            disease_info_dict_en["biological_control"] = [item.strip().lstrip('- ') for item in biological_match.group(1).strip().split('\n') if item.strip()]

    prevention_match = re.search(r"[*_]*Long-Term Prevention[:：]*[*_]*\s*(.+)", response_text_en, re.IGNORECASE | re.DOTALL)
    if prevention_match:
        disease_info_dict_en["prevention"] = [item.strip().lstrip('- ') for item in prevention_match.group(1).strip().split('\n') if item.strip()]

    logging.info("Parsed English disease information.")

    # Now, translate each extracted piece of information to the target language
    disease_info_dict_translated = {}
    if base_target_lang != 'en':
        for key, value in disease_info_dict_en.items():
            if isinstance(value, str):
                disease_info_dict_translated[key] = translate_text(value, base_target_lang, 'en')
            elif isinstance(value, list):
                translated_list = [translate_text(item, base_target_lang, 'en') for item in value]
                disease_info_dict_translated[key] = translated_list
        logging.info(f"Translated structured disease information to '{base_target_lang}'.")
    else:
        disease_info_dict_translated = disease_info_dict_en # If target is English, no translation needed

    return jsonify({"information": disease_info_dict_translated})


@app.route('/stt_process', methods=['POST'])
def stt_process():
    """
    Receives audio from frontend, sends to Google Cloud Speech-to-Text, returns transcript.
    """
    if speech_client is None:
        logging.error("Speech-to-Text client not initialized.")
        return jsonify({'error': 'Speech-to-Text service not available'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    language_code_full = request.form.get('language_code', 'en-US')
    base_language_code = language_code_full.split('-')[0] # e.g., 'en', 'hi'

    try:
        audio_content = audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS, # Assuming webm from MediaRecorder
            sample_rate_hertz=48000, # Common sample rate for webm/opus, adjust if needed
            language_code=language_code_full, # Use full language code for STT
            enable_automatic_punctuation=True,
        )

        logging.info(f"Sending audio for STT with language: {language_code_full}")
        response = speech_client.recognize(config=config, audio=audio)

        transcript = ""
        if response.results:
            # Get the most confident transcript
            transcript = response.results[0].alternatives[0].transcript
            logging.info(f"STT Transcript: '{transcript}'")
        else:
            logging.warning("STT returned no results.")
            transcript = "" # No speech detected or recognized

        return jsonify({'transcript': transcript})

    except Exception as e:
        logging.exception("Error during Speech-to-Text processing:")
        return jsonify({'error': str(e)}), 500

@app.route('/tts_process', methods=['POST'])
def tts_process():
    """
    Receives text and language, sends to Google Cloud Text-to-Speech, returns audio.
    """
    if tts_client is None:
        logging.error("Text-to-Speech client not initialized.")
        return jsonify({'error': 'Text-to-Speech service not available'}), 500

    data = request.json
    text = data.get('text', '')
    language_code_full = data.get('language_code', 'en-US')
    base_language_code = language_code_full.split('-')[0] # e.g., 'en', 'hi'

    if not text:
        return jsonify({'error': 'No text provided for TTS'}), 400

    try:
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request, select the language code and the voice gender
        # You might want to customize voice selection for better quality/variety
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code_full, # Use full language code for TTS
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # Or FEMALE/MALE
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        logging.info(f"Synthesizing speech for text '{text[:50]}...' in language: {language_code_full}")
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # The audio_content is base64-encoded binary data
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        return jsonify({'audio_content': audio_base64})

    except Exception as e:
        logging.exception("Error during Text-to-Speech processing:")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logging.info("Starting crop Disease Classification API Flask application.")
    app.run(debug=False, host='0.0.0.0', port=5000)




