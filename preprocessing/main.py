import yaml
import preprocessing.image_character.image_feature as img_features
import preprocessing.audio.create_audiofile as audio

if __name__ == '__main__':
    stream = open('preprocessing/config.yaml', 'r')
    config = yaml.load(stream)

    # img_features.extract(config["TRAILERS_FOLDER"], config["VIDEO_INFO_FOLDER"])
    audio.extract_audio(config["TRAILERS_FOLDER"], config["AUDIO_FOLDER"], config["AUDIO_INFO_FOLDER"])



# темп (быстрота смены цвета) по кадрам (берем насыщенность и яркость) - собрать среднее по каждому кадру, посмотреть на данные
# https: // dat602everyware.wordpress.com / 2016 / 10 / 16 / 55 /
# color mood classification - https://www.sciencedirect.com/topics/engineering/colour-difference-formula
# csv - https://docs.python.org/3/library/csv.html
# https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
# https://pyscenedetect.readthedocs.io/en/latest/examples/usage-example/#getting-started - done
# https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
# http://rstudio-pubs-static.s3.amazonaws.com/155921_d1b0d531118d46839a747b7c8b90e08b.html
# доминирующие цвета, освещенность, контрастность, глубина цвета, количество используемых цветов, переход из темных сцен в светлые
# корреляция музыки с визуальными характеристиками
# https://www.britannica.com/art/motion-picture/Qualities-of-the-film-image
# https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html
