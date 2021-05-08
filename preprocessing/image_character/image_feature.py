from os import listdir
from os.path import isfile, join, exists
import preprocessing.detect_scenes.detect as detect
from preprocessing.image_character import image_character as img_character
import csv


def process(file, filename, trailers_folder):
    print("start processing trailer file: ", file)
    img_filenames = detect.find_scenes(trailers_folder + file)
    csvfile = open(filename, 'w', newline='')
    fieldnames = ['scene', 'img', 'colors', 'temperature', 'brightness', 'colorfulness', 'hue']
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
                'hue': img_info['hue']
            })


def extract(trailers_folder, info_folder):
    trailer_files = [f for f in listdir(trailers_folder) if isfile(join(trailers_folder, f))]
    filtered_files = list(filter(lambda filename: filename[0] != '.', trailer_files))

    icount = 0
    for file in filtered_files:
        print(icount)
        filename = info_folder + file[:-4] + '.csv'
        try:
            csvfile = open(filename, 'r', newline='')
        except:
            print("no such file: ", filename)
            icount += 1
            continue
        if exists(filename):
            reader = csv.reader(csvfile)
            i = next(reader)
            print(i)
            lines = len(list(reader))
            if lines >= 10:
            # if i == ['scene', 'img', 'colors', 'temperature', 'brightness', 'colorfulness', 'hue']:
                print("already exist, pass")
                icount += 1
                continue
        process(file, filename, trailers_folder)
        icount += 1