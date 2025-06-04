import os
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model("speech_cnn_model.h5")
classes = ['yes', 'no', 'up', 'down', 'stop']
le = LabelEncoder()
le.fit(classes)

sample_rate = 16000
duration = 1
max_pad_len = 128

def extract_features_from_array(audio):
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    if melspec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec_db = melspec_db[:, :max_pad_len]
    return melspec_db

def record_audio():
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    recording = recording.flatten()
    print("Recording finished.")
    return recording

def predict_live(threshold=0.7):
    audio = record_audio()
    features = extract_features_from_array(audio)
    features = features[np.newaxis, ..., np.newaxis]
    prediction = model.predict(features)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    if confidence >= threshold:
        predicted_class = le.inverse_transform([class_idx])[0]
        print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
    else:
        print(f"Prediction uncertain (confidence: {confidence:.2f}). Likely silence or unknown.")

while True:
    predict_live()
    cont = input("Run again? (y/n): ")
    if cont.lower() != "y":
        break
