from os import listdir, remove
from os.path import isfile, join
import detect_scenes.detect as detect
import image_character.image_character as img_character
import csv


def extract(trailers_folder, info_folder):
    files = [f for f in listdir(trailers_folder) if isfile(join(trailers_folder, f))]
    filtered_files = list(filter(lambda filename: filename[0] != '.', files))
    print(filtered_files)
    for file in filtered_files:
        print("start processing trailer file: ", file)
        img_filenames = detect.find_scenes(trailers_folder + file)
        csvfile = open(info_folder + file[:-4] + '.csv', 'w', newline='')
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