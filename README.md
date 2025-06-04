
# Real-Time Speech Command Recognition using DSP + ML

###  Project Summary

This project implements a complete end-to-end DSP + ML pipeline for real-time speech command recognition.  
It combines:

- Audio recording from microphone
- Digital Signal Processing: STFT, Mel-spectrogram extraction
- Machine Learning: CNN classification
- Real-time inference with confidence thresholding
- Full signal-level debugging & visualization

---

### Target Use Case

Directly aligned with roles in:

- Apple's AIML Body-Sensing Intelligence Group
- ML for physiological signals
- DSP for sensor and time-series data

---

## Pipeline Overview

1️⃣ **Recording & Preprocessing**

- Sampling Rate: 16kHz  
- STFT → Mel Spectrogram  
- DSP Params: `n_fft=2048`, `hop_length=512`, `n_mels=64`

2️⃣ **CNN Model**

- Trained on 5 classes (`yes`, `no`, `up`, `down`, `stop`)  
- Speech Commands Dataset v0.02  
- Model achieves ~83-85% test accuracy.

3️⃣ **Real-Time Inference**

- Live microphone input
- Mel Spectrogram extracted on-the-fly
- Model prediction with confidence thresholding

4️⃣ **Visualization**

- Waveform inspection
- Spectrogram visualization for raw and dataset samples

---

## Files

| File | Description |
| ---- | ----------- |
| `train_model.py` | Model training pipeline |
| `realtime_inference.py` | Real-time live microphone prediction |
| `visualize_spectrogram.py` | Live spectrogram visualizer |
| `data_exploration.py` | Full DSP chain: waveform, replay, spectrogram, multi-class sample visualization |
| `speech_cnn_model.h5` | Saved trained model (produced after training) |
| `requirements.txt` | Dependencies list |

---

##  Installation

```bash
git clone <repo-url>
cd speech-dsp-ml-demo
pip install -r requirements.txt

