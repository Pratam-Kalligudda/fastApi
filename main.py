# from fastapi import FastAPI, UploadFile, File
# import io, librosa
# import numpy as np
# import tensorflow as tf
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# def preprocess_audio_mfcc(audio_file, target_length=50000, noise_level=0.001):
#     audio, sr = librosa.load(audio_file, sr=16000)
#     audio_smoothed = np.convolve(audio, np.ones(3) / 3, mode='same')
#     audio_normalized = audio_smoothed / np.max(np.abs(audio_smoothed))
#     if len(audio_normalized) < target_length:
#         audio_normalized = np.pad(audio_normalized, (0, target_length - len(audio_normalized)))
#     else:
#         audio_normalized = audio_normalized[:target_length]
#     mfccs = []
#     mfcc = librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13)
#     mfccs.append(mfcc)
#     mfccs = np.array(mfccs)
#     return mfccs


# def predict_result(aud):
#     mfcc = preprocess_audio_mfcc(audio_file=aud)
#     model = tf.keras.models.load_model('prediction_accuracy.keras')
#     predicted_accuracy_scores = model.predict(mfcc)
#     return float(predicted_accuracy_scores[0][0])


# # FastAPI
# app = FastAPI()

# @app.post("/upload/")
# async def create_upload_file(file: UploadFile = File(...)):
#     aud = io.BytesIO(await file.read())
#     res = predict_result(aud)
#     return {"result": res}



from fastapi import FastAPI, File, UploadFile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import azure.cognitiveservices.speech as speechsdk
import torch
import io
import threading
import librosa
import os
from dotenv import load_dotenv, dotenv_values 
app = FastAPI()

# Load the Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

load_dotenv() 
# Azure Speech Service configurations
speech_key = os.getenv("SPEECH_KEY")
service_region = os.getenv("REGION")

# Helper function to read the wave header
def read_wave_header(file):
    import wave
    with wave.open(file, 'rb') as wav_file:
        framerate = wav_file.getframerate()
        bits_per_sample = wav_file.getsampwidth() * 8
        num_channels = wav_file.getnchannels()
    return framerate, bits_per_sample, num_channels

# Function to push data into the audio stream
def push_stream_writer(stream, audio_bytes):
    chunk_size = 1024  # Chunk size for streaming
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        stream.write(chunk)
    stream.close()

# Pronunciation assessment function using an audio stream
def pronunciation_assessment_from_stream(audio_stream: io.BytesIO, reference_text: str):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    framerate, bits_per_sample, num_channels = read_wave_header(audio_stream)
    format = speechsdk.audio.AudioStreamFormat(samples_per_second=framerate, bits_per_sample=bits_per_sample, channels=num_channels)
    stream = speechsdk.audio.PushAudioInputStream(format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True)
    pronunciation_config.enable_prosody_assessment()

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language='en-US', audio_config=audio_config)
    pronunciation_config.apply_to(speech_recognizer)

    audio_bytes = audio_stream.read()
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream, audio_bytes])
    push_stream_writer_thread.start()
    result = speech_recognizer.recognize_once_async().get()
    push_stream_writer_thread.join()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        return {
            'text': result.text,
            'accuracy_score': pronunciation_result.accuracy_score,
            'prosody_score': pronunciation_result.prosody_score,
            'pronunciation_score': pronunciation_result.pronunciation_score,
            'completeness_score': pronunciation_result.completeness_score,
            'fluency_score': pronunciation_result.fluency_score,
            'words': [{'word': word.word, 'accuracy_score': word.accuracy_score, 'error_type': word.error_type} for word in pronunciation_result.words]
        }
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return {"error": "No speech could be recognized"}
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return {"error": f"Speech Recognition canceled: {cancellation_details.reason}", "details": cancellation_details.error_details}

# FastAPI endpoint to handle file uploads and perform both transcription and pronunciation assessment
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    audio_stream = io.BytesIO(await file.read())
    
    # Transcription using Wav2Vec2
    audio, sr = librosa.load(audio_stream, sr=16000)
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

    # Perform pronunciation assessment using the transcription as reference text
    result = pronunciation_assessment_from_stream(io.BytesIO(audio_stream.getvalue()), transcription)
    return {"result": result}
