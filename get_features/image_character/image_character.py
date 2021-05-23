import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import color_temp
import matplotlib.pyplot as plt


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar


def extract_color(image):
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=5) #KMeans with 6 threads
    clt.fit(image)

    hist = centroid_histogram(clt)
    # bar = plot_colors(hist, clt.cluster_centers_)
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()

    return [clt.cluster_centers_, hist]

def class_hue(h, hue):
    if 30 < h < 50:
        if not hasattr(hue, 'orange'):
            hue['orange'] = 1
        else:
            hue['orange'] = hue['orange'] + 1
        return hue
    if 50 < h < 70:
        if not hasattr(hue, 'yellow'):
            hue['yellow'] = 1
        else:
            hue['yellow'] = hue['yellow'] + 1
        return hue
    if 70 < h < 160:
        if not hasattr(hue, 'green'):
            hue['green'] = 1
        else:
            hue['green'] = hue['green'] + 1
        return hue
    if 160 < h < 200:
        if not hasattr(hue, 'aqua'):
            hue['aqua'] = 1
        else:
            hue['aqua'] = hue['aqua'] + 1
        return hue
    if 200 < h < 240:
        if not hasattr(hue, 'blue'):
            hue['blue'] = 1
        else:
            hue['blue'] = hue['blue'] + 1
        return hue
    if 240 < h < 300:
        if not hasattr(hue, 'violet'):
            hue['violet'] = 1
        else:
            hue['violet'] = hue['violet'] + 1
        return hue
    if 300 < h < 330:
        if not hasattr(hue, 'pink'):
            hue['pink'] = 1
        else:
            hue['pink'] = hue['pink'] + 1
        return hue
    else:
        if not hasattr(hue, 'red'):
            hue['red'] = 1
        else:
            hue['red'] = hue['red'] + 1
        return hue

def extract_img(image_filename):
    image = cv2.imread(image_filename)
    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    [colors, _] = extract_color(image)

    img_dict = {"colors": colors}
    temps = []
    for color in colors:
        temps.append(color_temp.rgb_to_temperature(color))

    img_dict["temps"] = np.mean(temps)

    brightness = []
    colorfulness = []
    hue = {}
    for color in colors:
        [h, s, v] = rgb_to_hsv(color[0], color[1], color[2])
        brightness.append(v)
        colorfulness.append(s)
        class_hue(h, hue)

    max = 0
    for attr, value in hue.items():
        if value > max:
            max = value
            img_dict["hue"] = attr

    img_dict["brightness"] = np.mean(brightness)
    img_dict["colorfulness"] = np.mean(colorfulness)

    return img_dict


