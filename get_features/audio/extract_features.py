import pandas as pd
import os
from IPython.display import display
from pydub import AudioSegment
import numpy as np
import librosa, librosa.display


def feature_extraction(y, sr):
    hop_length = 256
    frame_length = 512

    energy = np.array([
        sum(abs(y[i:i + frame_length] ** 2))
        for i in range(0, len(y), hop_length)
    ])
    rmse = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length, center=True)

    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    zero_crossings = librosa.zero_crossings(y, sr)
    tempo, beat_times = librosa.beat.beat_track(y, sr=sr, units='time')
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
    Y = librosa.stft(y)
    Ydb = librosa.amplitude_to_db(abs(Y))
    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    return {
        'energy': np.mean(energy),
        'tempo': tempo,
        'zero_cross': np.mean(zero_crossings),
        'beat_times': np.mean(beat_times),
        'harmonic': np.mean(y_harmonic),
        'percussive': np.mean(y_percussive),
        'amplitude': np.mean(Ydb),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'mfcc': np.mean(mfcc),
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spec_bw': np.mean(spec_bw)
    }


def process_every_five(array):
    new_arr = []
    sum = 0
    for i in range(len(array)):
        if (i % 5 != 0):
            sum += array[i]
        else:
            new_arr.append(sum)
            sum = 0

    return new_arr


def normalize(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    display(normalized_df.head())
    return df


def extract_audio_features(AUDIO_FOLDER, VIDEO_INFO_FOLDER):
    film_csv = "/Volumes/Seagate/natasha-diploma/content_data.csv"
    trailers_csv = "/Volumes/Seagate/natasha-diploma/trailers.csv"

    films = pd.read_csv(film_csv, index_col=None, header=0)
    display(films.head())

    trailers_meta = pd.read_csv(trailers_csv, index_col=None, header=0)
    display(trailers_meta.head())

    for i in range(0, len(trailers_meta)):
        name = trailers_meta.iloc[i]['trailers_name']
        img_info = VIDEO_INFO_FOLDER + '/' + name + '.csv'
        audio = AUDIO_FOLDER + '/' + name + '.wav'

        if os.path.exists(audio) and os.path.exists(img_info):
            print(name, i)
            x, sr = librosa.load(audio)
            audio_duration = librosa.get_duration(x) * 1000

            img_df = pd.read_csv(img_info, index_col=None, header=0)
            scenes = img_df['scene'].iloc[-1]
            dur_per_scene = audio_duration / (scenes + 1)

            if 'tempo' in img_df:
                print("pass file: ", img_info)
                continue

            else:
                energy = []
                tempo = []
                zero_cross = []
                harmonic = []
                percussive = []
                amplitude = []
                spectral_rolloff = []
                mfcc = []
                chroma_stft = []
                rmse = []
                spec_bw = []
                newAudio = AudioSegment.from_wav(audio)
                for i in range(0, len(img_df)):
                    scene = int(img_df.iloc[i]['scene'])
                    if (dur_per_scene * scene >= audio_duration):
                        break
                    newAudio_temp = newAudio[dur_per_scene * scene:dur_per_scene * (scene + 1)]
                    newAudio_temp.export('newSong.wav', format="wav")

                    x_temp, sr_temp = librosa.load('newSong.wav')

                    features = feature_extraction(x_temp, sr_temp)

                    energy.append(features['energy'])
                    tempo.append(features['tempo'])
                    zero_cross.append(features['zero_cross'])
                    harmonic.append(features['harmonic'])
                    percussive.append(features['percussive'])
                    amplitude.append(features['amplitude'])
                    spectral_rolloff.append(features['spectral_rolloff'])
                    mfcc.append(features['mfcc'])
                    chroma_stft.append(features['chroma_stft'])
                    rmse.append(features['rmse'])
                    spec_bw.append(features['spec_bw'])
                img_df['energy'] = energy
                img_df['tempo'] = tempo
                img_df['zero_cross'] = zero_cross
                img_df['harmonic'] = harmonic
                img_df['percussive'] = percussive
                img_df['amplitude'] = amplitude
                img_df['spectral_rolloff'] = spectral_rolloff
                img_df['mfcc'] = mfcc
                img_df['chroma_stft'] = chroma_stft
                img_df['rmse'] = rmse
                img_df['spec_bw'] = spec_bw

                os.remove('newSong.wav')
                img_df.to_csv(img_info)
                print("done with", img_info)


