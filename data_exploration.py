# import os
# import numpy as np
# import librosa
# import librosa.display
# import sounddevice as sd
# import matplotlib.pyplot as plt
# import random
# import sounddevice as sd

# # DSP Parameters
# sample_rate = 16000
# duration = 1
# n_fft = 2048
# hop_length = 512
# n_mels = 64
# max_pad_len = 128

# # Dataset Path
# DATASET_PATH = r"C:\Users\asrit\OneDrive - Arizona State University\Personal Proects\ASU_applications\dsp\speech_dataset"
# classes = ['yes', 'no', 'up', 'down', 'stop']

# # 1️⃣ RECORD AUDIO + SHOW WAVEFORM
# def record_audio():
#     print("Recording...")
#     recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     print("Recording finished.")
#     return recording.flatten()

# audio = record_audio()

# plt.figure(figsize=(10, 3))
# plt.plot(np.linspace(0, duration, len(audio)), audio)
# plt.title("Recorded Audio Waveform")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()

# # 2️⃣ MEL SPECTROGRAM OF RECORDED AUDIO (with full DSP params)

# melspec = librosa.feature.melspectrogram(
#     y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
# )
# melspec_db = librosa.power_to_db(melspec, ref=np.max)

# plt.figure(figsize=(10, 4))
# librosa.display.specshow(melspec_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
# plt.title(f"Mel Spectrogram (n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})")
# plt.colorbar(format="%+2.0f dB")
# plt.tight_layout()
# plt.show()

# # 3️⃣ SPECTROGRAMS FROM DIFFERENT CLASSES

# plt.figure(figsize=(15, 8))

# for i, label in enumerate(classes):
#     folder = os.path.join(DATASET_PATH, label)
#     files = os.listdir(folder)
#     file_path = os.path.join(folder, random.choice(files))

#     audio, sr = librosa.load(file_path, sr=sample_rate)
#     melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#     melspec_db = librosa.power_to_db(melspec, ref=np.max)

#     plt.subplot(2, 3, i+1)
#     librosa.display.specshow(melspec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
#     plt.title(f"{label}")
#     plt.colorbar(format="%+2.0f dB")

# plt.tight_layout()
# plt.show()

import os
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
import random

# DSP Parameters
sample_rate = 16000
duration = 1  # 1 second recording
n_fft = 2048
hop_length = 512
n_mels = 64
max_pad_len = 128

# Dataset Path (change this to your dataset path if needed)
DATASET_PATH = r"C:\Users\asrit\OneDrive - Arizona State University\Personal Proects\ASU_applications\dsp\speech_dataset"
classes = ['yes', 'no', 'up', 'down', 'stop']

# 1️⃣ RECORD AUDIO + PLAYBACK + SHOW WAVEFORM
def record_audio():
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    
    # Playback
    print("Playing back...")
    sd.play(recording, samplerate=sample_rate)
    sd.wait()
    
    return recording.flatten()

audio = record_audio()

# Plot waveform
plt.figure(figsize=(10, 3))
plt.plot(np.linspace(0, duration, len(audio)), audio)
plt.title("Recorded Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 2️⃣ MEL SPECTROGRAM OF RECORDED AUDIO (with full DSP params)
melspec = librosa.feature.melspectrogram(
    y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
melspec_db = librosa.power_to_db(melspec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.title(f"Mel Spectrogram (n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels})")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()

# 3️⃣ SPECTROGRAMS FROM DIFFERENT CLASSES
plt.figure(figsize=(15, 8))

for i, label in enumerate(classes):
    folder = os.path.join(DATASET_PATH, label)
    files = os.listdir(folder)
    file_path = os.path.join(folder, random.choice(files))

    class_audio, sr = librosa.load(file_path, sr=sample_rate)
    class_melspec = librosa.feature.melspectrogram(
        y=class_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    class_melspec_db = librosa.power_to_db(class_melspec, ref=np.max)

    plt.subplot(2, 3, i+1)
    librosa.display.specshow(class_melspec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title(f"{label}")
    plt.colorbar(format="%+2.0f dB")

plt.tight_layout()
plt.show()
