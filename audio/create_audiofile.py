import moviepy.editor as mp

TRAILERS_FOLDER = "/Users/nutochkina/Documents/projects/trailers_mp4/"
AUDIO_FOLDER = "/Users/nutochkina/Documents/projects/audio_wav/"


def extract_audio(files):
    for file in files:
        video_path = TRAILERS_FOLDER + file
        my_clip = mp.VideoFileClip(video_path)
        my_clip.audio.write_audiofile(AUDIO_FOLDER + file[:-3] + "wav")
