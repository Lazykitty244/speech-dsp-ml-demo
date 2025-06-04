import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt

sample_rate = 16000
duration = 1
max_pad_len = 128

def record_audio():
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return recording.flatten()

def visualize():
    audio = record_audio()
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(melspec_db, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

visualize()
