from imageai.Detection import ObjectDetection
from os import listdir, getcwd
from os.path import isfile, join


def get_objects_from_scene(trailers_folder):
    trailer_files = [f for f in listdir(trailers_folder) if isfile(join(trailers_folder, f))]
    filtered_files = list(filter(lambda filename: filename[0] != '.', trailer_files))

    file_scenes = []
    for file in filtered_files:
        scenes = [f for f in listdir(trailers_folder + '/' + file[:-4] + '/')
                  if isfile(join(trailers_folder + '/' + file[:-4] + '/', f))]

        objects = []
        for scene in scenes:
            execution_path = getcwd()

            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(join(execution_path + "/preprocessing/find_objects/", "resnet50_coco_best_v2.1.0.h5"))
            detector.loadModel()
            print(trailers_folder + file[:-4] + '/' + scene)
            detections = detector.detectObjectsFromImage(
                input_image=join(trailers_folder + file[:-4] + '/', scene),
                output_image_path=join(execution_path + "/preprocessing/find_objects/", "new.jpeg")
            )

            print(detections)

            for eachObject in detections:
                print(eachObject["name"], " : ", eachObject["percentage_probability"])
                if eachObject["percentage_probability"] > 50:
                    objects.append(eachObject["name"])

        file_scenes.append(objects)



