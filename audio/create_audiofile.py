import moviepy.editor as mp


def extract_audio(files, trailers_folder, audio_folder):
    for file in files:
        video_path = trailers_folder + file
        my_clip = mp.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(audio_folder + file[:-3] + "wav")
