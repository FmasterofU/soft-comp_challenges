import os
import numpy as np
import cv2  # OpenCV
from sklearn.svm import SVC  # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import math

# import libraries here


face_detector = dlib.get_frontal_face_detector()
face_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
nbins = 9
cell_size = (8, 8)
block_size = (3, 3)
face_image_width = 200
face_image_height = 200
hog = cv2.HOGDescriptor(_winSize=(face_image_width // cell_size[1] * cell_size[1],
                                  face_image_height // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)


def load_rgb_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def calc_in_image_rectangle(rect, image):
    x, y, w, h = rect
    if y < 0:
        h = h + y
        y = 0
    if x < 0:
        w = w + x
        x = 0
    if x + w >= image.shape[1]:
        w = image.shape[1] - x - 1
    if y + h >= image.shape[0]:
        h = image.shape[0] - y - 1
    rect = (x, y, w, h)
    return rect


def face_descriptor(image, image_path):
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        None, None, None, None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = face_detector(gray, 1)
    roi_index = None if len(rects) == 0 else sorted([i for i in range(len(rects))],
                                                    key=lambda x: math.pow(face_utils.rect_to_bb(rects[x])[2], 2) +
                                                                  math.pow(face_utils.rect_to_bb(rects[x])[3], 2),
                                                    reverse=True)[0]
    print(rects, image_path, roi_index)
    face_rect = None
    if roi_index is None:
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        face_rect = (0, 0, image.shape[1], image.shape[0])
    else:
        face_landmarks = face_utils.shape_to_np(face_shape_predictor(gray, rects[roi_index]))
        rect = face_utils.rect_to_bb(rects[roi_index])
        face_rect = calc_in_image_rectangle(rect, image)
    face_image = image[face_rect[1]:face_rect[1] + face_rect[3] + 1, face_rect[0]:face_rect[0] + face_rect[2] + 1, :]

    face_hog = hog.compute(resize_image_channel(cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)))

    hsl_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HLS)
    _, hue_histogram = histogram(hsl_face[:, :, 0], 180)
    _, saturation_histogram = histogram(hsl_face[:, :, 2], 255)
    _, value_histogram = histogram(hsl_face[:, :, 1], 255)

    return face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram


def minimize_landmark_coordinates(face_landmarks):
    y_min = min(face_landmarks, key=lambda x: x[0])[0]
    x_min = min(face_landmarks, key=lambda x: x[1])[1]
    face_landmarks = face_landmarks - np.array([y_min, x_min])
    return face_landmarks


def resize_image_channel(channel, size=(face_image_height, face_image_width)):
    resized_channel = cv2.resize(channel, size, interpolation=cv2.INTER_NEAREST)
    return resized_channel


def histogram(channel, hist_x_max):
    height, width = channel.shape[0:2]
    x = range(0, hist_x_max + 1)
    y = np.zeros(hist_x_max + 1)
    for i in range(0, height):
        for j in range(0, width):
            pixel = channel[i, j]
            y[pixel] += 1
    return x, y


def normalize(array):
    return array/np.amax(array)


def train_or_load_age_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = None
    try:
        model = load('knn_age.joblib')
    except FileNotFoundError as e:
        knn = KNeighborsClassifier(n_neighbors=len(train_image_labels))
        x_train = []
        for i in range(len(train_image_paths)):
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            x_train.append(face_hog.flatten())
        x_train = np.array(x_train)
        y_train = np.array(train_image_labels) // 5
        print(y_train.shape, x_train.shape)
        print(y_train)
        knn = knn.fit(x_train, y_train)
        model = knn
        dump(knn, 'knn_age.joblib')
        print("Train accuracy: ", accuracy_score(y_train, knn.predict(x_train)))
    return model


def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = None
    try:
        model = load('svm_gender.joblib')
    except FileNotFoundError as e:
        svm = SVC(kernel='linear', probability=True)
        x_train = []
        for i in range(len(train_image_paths)):
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            if face_landmarks is None:
                print(i)
                train_image_labels.pop(i)
                continue
            face_landmarks = minimize_landmark_coordinates(face_landmarks)
            x_train.append(face_landmarks.flatten())
        x_train = np.array(x_train)
        y_train = np.array(train_image_labels)
        print(y_train.shape, x_train.shape)
        print(y_train)
        svm.fit(x_train, y_train)
        model = svm
        dump(svm, 'svm_gender.joblib')
        print("Train accuracy: ", accuracy_score(y_train, svm.predict(x_train)))
    return model


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    model = None
    try:
        model = load('knn_race.joblib')
    except FileNotFoundError as e:
        knn = KNeighborsClassifier(n_neighbors=len(train_image_labels))
        x_train = []
        for i in range(len(train_image_paths)):
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            x = []
            x.extend(hue_histogram.flatten().tolist())
            x.extend(saturation_histogram.flatten().tolist())
            x.extend(value_histogram.flatten().tolist())
            x = np.array(x)
            x_train.append(x.flatten())
        x_train = np.array(x_train)
        y_train = np.array(train_image_labels)
        print(y_train.shape, x_train.shape)
        print(y_train)
        knn = knn.fit(x_train, y_train)
        model = knn
        dump(knn, 'knn_race.joblib')
        print("Train accuracy: ", accuracy_score(y_train, knn.predict(x_train)))
    return model


def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    age = 150000000  # 0
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    image = load_rgb_image(image_path)
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        face_descriptor(image, image_path)
    temp = trained_model.predict(np.array([face_hog.flatten()]))[0]
    print(temp)
    age = temp * 5
    print(age)
    return age


def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    gender = 2  # 1
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    return gender
    image = load_rgb_image(image_path)
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        face_descriptor(image, image_path)
    if face_landmarks is None:
        return 2
    face_landmarks = minimize_landmark_coordinates(face_landmarks)
    temp = trained_model.predict(np.array([face_landmarks.flatten()]))
    print(temp)
    gender = int(temp[0])
    print(gender)
    return gender


def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    race = 5  # 4
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    return race
    image = load_rgb_image(image_path)
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        face_descriptor(image, image_path)
    x = []
    x.extend(hue_histogram.flatten().tolist())
    x.extend(saturation_histogram.flatten().tolist())
    x.extend(value_histogram.flatten().tolist())
    x = np.array(x)
    temp = trained_model.predict(np.array([x.flatten()]))
    print(temp)
    race = int(temp[0])
    print(race)
    return race
