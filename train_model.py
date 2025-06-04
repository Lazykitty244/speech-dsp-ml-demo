import os
import numpy as np
import librosa
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Dataset Path
DATASET_PATH = r"C:\Users\asrit\OneDrive - Arizona State University\Personal Proects\ASU_applications\dsp\speech_dataset"

# Select classes
classes = ['yes', 'no', 'up', 'down', 'stop']
sample_rate = 16000
max_pad_len = 128

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    if melspec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
    else:
        melspec_db = melspec_db[:, :max_pad_len]
    return melspec_db

X, y = [], []
for label in classes:
    folder = os.path.join(DATASET_PATH, label)
    files = os.listdir(folder)
    files = random.sample(files, 500)
    for file in files:
        try:
            feature = extract_features(os.path.join(folder, file))
            X.append(feature)
            y.append(label)
        except:
            pass

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
model.evaluate(X_test, y_test)
model.save("speech_cnn_model.h5")
print("✅ Model trained and saved as speech_cnn_model.h5")
