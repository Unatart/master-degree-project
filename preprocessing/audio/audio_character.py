import librosa
import csv
import numpy as np


def process_audio(audio_path, info_path, audio_name):
    y, sr = librosa.load(audio_path)
    centro = librosa.feature.spectral_centroid(y, sr)[0]
    zero_crossings = librosa.zero_crossings(y, sr)
    zero_crossings_count = 0
    for i in range(len(zero_crossings)):
        if zero_crossings[i] == True:
            zero_crossings_count += 1
    tempo, beat_times = librosa.beat.beat_track(y, sr=sr, units='time')
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    Y = librosa.stft(y)
    Ydb = librosa.amplitude_to_db(abs(Y))
    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    csvfile = open(info_path + audio_name + 'csv', 'w', newline='')
    fieldnames = ['tempo',
                  'centro',
                  'zero_crossings',
                  'zero_cross_count',
                  'beat_times',
                  'harmonic',
                  'percussive',
                  'amplitude',
                  'spectral_rolloff',
                  'mfcc',
                  'chroma_stft',
                  'rmse',
                  'spec_bw']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        'tempo': tempo,
        'centro': centro,
        'zero_crossings': zero_crossings,
        'zero_cross_count': zero_crossings_count,
        'beat_times': beat_times,
        'harmonic': np.mean(y_harmonic),
        'percussive': np.mean(y_percussive),
        'amplitude': np.mean(Ydb),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'mfcc': np.mean(mfcc),
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spec_bw': np.mean(spec_bw)
    })
