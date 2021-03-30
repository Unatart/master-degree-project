from os import listdir, remove
from os.path import isfile, join
import detect_scenes.detect as detect
import image_character.image_character as img_character
import csv

TRAILERS_FOLDER = "/Users/nutochkina/Documents/projects/trailers_mp4/"
VIDEO_INFO_FOLDER = "/Users/nutochkina/Documents/projects/videoinfo/"

if __name__ == '__main__':
    files = [f for f in listdir(TRAILERS_FOLDER) if isfile(join(TRAILERS_FOLDER, f))]

    for file in files:
        print("start processing trailer file: ", file)
        img_filenames = detect.find_scenes(TRAILERS_FOLDER + file)
        csvfile = open(VIDEO_INFO_FOLDER + file[:-4] + '.csv', 'w', newline='')
        fieldnames = ['scene', 'img', 'colors', 'temperature', 'brightness', 'colorfulness', 'color_hist']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_cut in img_filenames:
            for img in img_filenames[img_cut]:
                img_info = img_character.extract_img(img)
                writer.writerow({
                    'scene': img_cut,
                    'img': img,
                    'colors': img_info['colors'],
                    'temperature': img_info['temps'],
                    'brightness': img_info['brightness'],
                    'colorfulness': img_info['colorfulness'],
                    'color_hist': img_info['color_hist']
                })

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
