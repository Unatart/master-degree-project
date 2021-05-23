import yaml
import get_features.image_character.image_feature as img_features
import get_features.audio as audio
import get_features.find_objects.find_object as find_objects
import argparse
import model.hrnn_model as model

if __name__ == '__main__':
    stream = open('config.yaml', 'r')
    config = yaml.load(stream)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sessions", help="файл с сессиями пользователя")
    parser.add_argument("-i", "--scenes_info", help="папка с данными сцен каждого трейлера")
    parser.add_argument("-m", "--movie_info", help="файл с данными о фильме трейлера")
    parser.add_argument("-o", "--operation", help="0 - деление трейлеров на сцены и выделение зрительных характеристик,"
                                            "1 - выделение аудио-характеристик,"
                                            "2 - получение объектов на сцене,"
                                            "3 - обучение,"
                                            "4 - предсказание")
    args = parser.parse_args()

    if args.operation == '0':
        img_features.extract(config["TRAILERS_FOLDER"], config["VIDEO_INFO_FOLDER"])
    if args.operation == '1':
        audio.create_audiofile.extract_audio(config["TRAILERS_FOLDER"], config["AUDIO_FOLDER"], config["AUDIO_INFO_FOLDER"])
        audio.extract_features.extract_audio_features(config["AUDIO_FOLDER"], config["VIDEO_INFO_FOLDER"])
    if args.operation == '2':
        find_objects.get_objects_from_scene(config["TRAILERS_FOLDER"])
    if args.operation == '3':
        model.train_model()
    if args.operation == '4':
        model.evaluate_and_predict()

