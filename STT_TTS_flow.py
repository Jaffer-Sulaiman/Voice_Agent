import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()


#STT imports
import whisper

#TTS imports
from google.cloud import texttospeech
import pyttsx3

#

def whisper_speech_to_text (audio_file_path):
    model = whisper.load_model("base")
    result = whisper.transcribe(model=model, audio=audio_file_path)

    with open ("./audio_files/test_audio1_transcription.txt", "w") as f:
        f.write(result["text"])

    return  result["text"]

def google_text_to_speech(ai_text):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=ai_text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # The response's audio_content is binary.
    with open("./voice_test_outputs/output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
    
    return 'Audio content written to file "output.mp3"'


def espeak_text_to_speech(ai_text):
    engine = pyttsx3.init()
    engine.save_to_file(ai_text , './voice_test_outputs/test.mp3')
    engine.runAndWait()
    return 'Audio content written to file "test.mp3"'


def coqui_xttsv2_text_to_speech(ai_text):
    pass
