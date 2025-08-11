import whisper

model = whisper.load_model("base")
result = whisper.transcribe(model=model, audio="./audio_files/test_audio1.mp3")

with open ("./audio_files/test_audio1_transcription.txt", "w") as f:
    f.write(result["text"])

