from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.scene_manager import save_images
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=65.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list()

    image_name_template = video_path[:-4] + '/scenedetect.tempfile.$SCENE_NUMBER.$IMAGE_NUMBER'
    image_filenames = save_images(
        scene_list=scenes,
        video_manager=video_manager,
        num_images=3,
        image_extension='jpg',
        image_name_template=image_name_template)

    return image_filenames

