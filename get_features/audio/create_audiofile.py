import moviepy.editor as mp
from os import listdir
from os.path import isfile, join


def extract_audio(trailers_folder, audio_folder, info_folder):
    files = [f for f in listdir(trailers_folder) if isfile(join(trailers_folder, f))]
    filtered_files = list(filter(lambda filename: filename[0] != '.', files))
    for file in filtered_files:
        video_path = trailers_folder + file
        my_clip = mp.VideoFileClip(video_path)
        audio_path = audio_folder + file[:-3] + "wav"
        my_clip.audio.write_audiofile(audio_path)