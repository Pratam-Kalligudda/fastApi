from fastapi import FastAPI, UploadFile, File
import io, librosa
import numpy as np
import tensorflow as tf


def preprocess_audio_mfcc(audio_file, target_length=50000, noise_level=0.001):
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_smoothed = np.convolve(audio, np.ones(3) / 3, mode='same')
    audio_normalized = audio_smoothed / np.max(np.abs(audio_smoothed))
    if len(audio_normalized) < target_length:
        audio_normalized = np.pad(audio_normalized, (0, target_length - len(audio_normalized)))
    else:
        audio_normalized = audio_normalized[:target_length]
    mfccs = []
    mfcc = librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13)
    mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    return mfccs


def predict_result(aud):
    mfcc = preprocess_audio_mfcc(audio_file=aud)
    model = tf.keras.models.load_model('prediction_accuracy.keras')
    predicted_accuracy_scores = model.predict(mfcc)
    return float(predicted_accuracy_scores[0][0])


# FastAPI
app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    aud = io.BytesIO(await file.read())
    res = predict_result(aud)
    return {"result": res}
