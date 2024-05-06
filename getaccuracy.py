import io
import librosa
import numpy as np
import tensorflow as tf
def load_model():
    # Load your pre-trained model (replace with your actual model loading logic)
    model = tf.keras.models.load_model('prediction_accuracy.keras')
    return model

def preprocess_audio_mfcc(audio_file, target_length=50000, noise_level=0.001):
    # Load audio data
    
    audio, sr = librosa.load(audio_file, sr=16000)
    # Noise reduction (simple moving average filter)
    audio_smoothed = np.convolve(audio, np.ones(3) / 3, mode='same')

    # Data augmentation (optional)
    # ... (implement if needed)

    # Normalize audio
    audio_normalized = audio_smoothed / np.max(np.abs(audio_smoothed))

    # Pad or truncate to the target length
    if len(audio_normalized) < target_length:
        audio_normalized = np.pad(audio_normalized, (0, target_length - len(audio_normalized)))
    else:
        audio_normalized = audio_normalized[:target_length]
    mfccs = []
    # Feature extraction (MFCC)
    mfcc = librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13)
    mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    return mfccs


# audio_file = "C:\\Users\\kalli\\project\\fr\\clips\\common_voice_fr_39586342.mp3"
file_path = "C:\\Users\\kalli\\project\\fr\\clips\\common_voice_fr_39586341.mp3"

# Open the file in binary read mode
with open(file_path, 'rb') as file:
  # Read the entire file content into a byte array
  audio_data = file.read()

audio_stream = io.BytesIO(audio_data)


mfcc = preprocess_audio_mfcc(audio_file=audio_stream)
model = load_model()

predicted_accuracy_scores = model.predict(mfcc)
# audio_stream = io.BytesIO(audio_file)
print(predicted_accuracy_scores[0][0])