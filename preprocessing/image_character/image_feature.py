from os import listdir
from os.path import isfile, join, exists
import preprocessing.detect_scenes.detect as detect
from preprocessing.image_character import image_character as img_character
import csv


def extract(trailers_folder, info_folder):
    trailer_files = [f for f in listdir(trailers_folder) if isfile(join(trailers_folder, f))]
    filtered_files = list(filter(lambda filename: filename[0] != '.', trailer_files))
    info_files = [f for f in listdir(info_folder) if isfile(join(info_folder, f))]
    print(info_files)
    files = list(set(filtered_files) - set(info_files))
    print(files)
    for file in files:
        # try:
            img_filenames = detect.find_scenes(trailers_folder + file)
            filename = info_folder + file[:-4] + '.csv'
            try:
                csvfile = open(filename, 'r', newline='')
            except:
                print("no such file: ", filename)
            if exists(filename):
                reader = csv.reader(csvfile)
                lines = len(list(reader))
                img_len = len(img_filenames) * len(img_filenames[0])
                if lines >= img_len:
                    print("already exist, pass")
                    continue
            print("start processing trailer file: ", file)
            csvfile = open(filename, 'w', newline='')
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
        # except:
        #     print('error with file ' + info_folder + file[:-4] + '.csv')