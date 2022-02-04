import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')
list_labels = np.load('list_labels.npy')

sampling_rate = 44100
duration = 4
sample_length = sampling_rate * duration

def get_mel_spectrogram(filename):
    data, sr = librosa.load(filename, sr=sampling_rate)
    if 0 < len(data):
        data, _ = librosa.effects.trim(data)
    if len(data) > sample_length:
        data = data[0:sample_length]
    else:
        padding = sample_length - len(data)
        offset = padding // 2
        data = np.pad(data, (offset, sample_length - len(data) - offset), 'constant')

    mel_spectrogram = librosa.feature.melspectrogram(data, sr=sampling_rate, n_fft=2048, hop_length=512, n_mels=80)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    return log_mel_spectrogram

def normalize(X):
    eps = 0.001
    if np.std(X) != 0:
        X_rescale = (X - np.mean(X)) / np.std(X)
    else:
        X_rescale = (X - np.mean(X)) / eps
    return np.array(X_rescale)

def prediction(file):
    mel_spectrogram = get_mel_spectrogram(file)
    x = normalize(mel_spectrogram)
    x = x.reshape(1,80,345,1)
    predicted = model.predict(x) 
    pred_label = list_labels[np.argsort(-predicted, axis=1)[:, :1]]
    prediction_label = pred_label[0][0]
    return prediction_label
