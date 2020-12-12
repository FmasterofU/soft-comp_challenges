import os
import numpy as np
import cv2  # OpenCV
from imblearn.under_sampling import ClusterCentroids
from keras.layers import Dense, Dropout
from keras.models import model_from_json, Sequential
from keras.optimizers import SGD
from sklearn.svm import SVC, SVR  # SVM klasifikator
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
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler

# import libraries here


face_detector = dlib.get_frontal_face_detector()
face_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
nbins = 9
cell_size = (32, 32)
block_size = (4, 4)
face_image_width = 200
face_image_height = 200
hog = cv2.HOGDescriptor(_winSize=(face_image_width // cell_size[1] * cell_size[1],
                                  face_image_height // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
ros = RandomOverSampler(random_state=0)
cc = ClusterCentroids(random_state=0)


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

    face_rect_landmarks = None if face_landmarks is None else face_landmarks - np.array([face_rect[1], face_rect[0]])
    #face_rect_landmarks = None
    hsl_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2HLS)
    _, hue_histogram = normalize(face_histogram(hsl_face[:, :, 0], 180, face_rect_landmarks))
    _, saturation_histogram = normalize(face_histogram(hsl_face[:, :, 2], 255, face_rect_landmarks))
    _, value_histogram = normalize(face_histogram(hsl_face[:, :, 1], 255, face_rect_landmarks))

    return face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram


def minimize_landmark_coordinates(face_landmarks):
    y_min = min(face_landmarks, key=lambda x: x[0])[0]
    x_min = min(face_landmarks, key=lambda x: x[1])[1]
    face_landmarks = face_landmarks - np.array([y_min, x_min])
    return face_landmarks


def quadratic_relative_landmark_distances(face_landmarks):
    y_min = min(face_landmarks, key=lambda x: x[0])[0]
    x_min = min(face_landmarks, key=lambda x: x[1])[1]
    y_max = max(face_landmarks, key=lambda x: x[0])[0]
    x_max = max(face_landmarks, key=lambda x: x[1])[1]
    diag = math.sqrt(math.pow(y_max - y_min, 2) + math.pow(x_max - x_min, 2))
    diag = 1
    distances = []
    for i in range(face_landmarks.shape[0]):
        for j in range(i, face_landmarks.shape[0]):
            distances.append(math.sqrt(math.pow(face_landmarks[i, 0] - face_landmarks[j, 0], 2) +
                                       math.pow(face_landmarks[i, 1] - face_landmarks[j, 1], 2)) / diag)
    return np.array(distances)


def resize_image_channel(channel, size=(face_image_height, face_image_width)):
    resized_channel = cv2.resize(channel, size, interpolation=cv2.INTER_NEAREST)
    return resized_channel


def face_histogram(channel, hist_x_max, face_landmarks=None):
    face_hull = None if face_landmarks is None else cv2.convexHull(face_landmarks)

    def is_pixel_in_hull(y, x, face_hull):
        #print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        check = cv2.pointPolygonTest(face_hull, (y, x), False) > 0
        #print(check)
        return 1 if check else 0

    height, width = channel.shape[0:2]
    x = range(0, hist_x_max + 1)
    y = np.zeros(hist_x_max + 1)
    for i in range(0, height):
        for j in range(0, width):
            pixel = channel[i, j]
            y[pixel] += 1 if face_hull is None else is_pixel_in_hull(i, j, face_hull)
    return x, y


def normalize(array):
    return array/np.amax(array)


def sum_normalization(array):
    return array#/np.sum(array)


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

    model = [None, None]
    return model
    try:
        model[0] = load('svm_age.joblib')
    except FileNotFoundError as e:
        knn = SVR(kernel='linear')#, probability=True)#, class_weight='balanced')#KNeighborsClassifier(n_neighbors=len(train_image_labels))
        x_train = []
        for i in range(len(train_image_paths)):
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            x_train.append(face_hog.flatten())
        x_train = np.array(x_train)
        y_train = np.array(train_image_labels) // 5
        print(y_train.shape, x_train.shape)
        #x_train, y_train = ros.fit_resample(x_train, y_train)
        print(y_train.shape, x_train.shape)
        print(y_train)
        knn = knn.fit(x_train, y_train)
        model[0] = knn
        dump(knn, 'svm_age.joblib')
        print("Train accuracy: ", accuracy_score(y_train, knn.predict(x_train)))

    nnmodel = None
    try:
        with open(os.path.join(os.getcwd(), 'AgeNeuralNetParams.json'), 'r') as nnp_file:
            nnmodel = model_from_json(nnp_file.read())
        nnmodel.load_weights(os.path.join(os.getcwd(), 'AgeNeuralNetWeights.h5'))
    except Exception as e:
        nn = Sequential()
        nn.add(Dense(512, input_dim=1296, activation='sigmoid'))
        nn.add(Dropout(0.15))
        # nn.add(Dense(256, activation='sigmoid'))
        # nn.add(Dropout(0.3))
        nn.add(Dense(192, activation='sigmoid'))
        #nn.add(Dropout(0.3))
        nn.add(Dense(24, activation='softmax'))
        y_list = y_train.tolist()
        y = np.array(np.zeros((len(y_list), 24)), np.float32)
        for i in range(len(y_list)):
            y[i, y_list[i]] = 1
        print(x_train.shape, y.shape)
        print(x_train[0], y[0])
        sgd = SGD(lr=0.3, momentum=0.9)
        nn.compile(loss='mean_squared_error', optimizer=sgd)
        nn.fit(x_train, y, epochs=1000, batch_size=1, verbose=1, shuffle=False)
        nnmodel = nn
        params = nnmodel.to_json()
        try:
            with open(os.path.join(os.getcwd(), 'AgeNeuralNetParams.json'), 'w') as nnp_file:
                nnp_file.write(params)
            nnmodel.save_weights(os.path.join(os.getcwd(), 'AgeNeuralNetWeights.h5'))
        except Exception as e:
            print(e)
            pass
    model[1] = nnmodel
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

    model = [None, None]
    x_train = []
    y_train = []
    try:
        model[0] = load('svm_gender.joblib')
    except FileNotFoundError as e:
        svm = SVC(kernel='linear', probability=True)
        x_train = []
        y_train = []
        for i in range(len(train_image_paths)):
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            if face_landmarks is None:
                print("Neprihvatljivi pol")
                continue
            y_train.append(train_image_labels[i])
            #face_landmarks = minimize_landmark_coordinates(face_landmarks)
            face_landmarks = quadratic_relative_landmark_distances(face_landmarks)
            x_train.append(face_landmarks.flatten())
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(y_train.shape, x_train.shape)
        x_train, y_train = ros.fit_resample(x_train, y_train)
        print(y_train.shape, x_train.shape)
        print(y_train)
        svm.fit(x_train, y_train)
        model[0] = svm
        dump(svm, 'svm_gender.joblib')
        #print("Train accuracy: ", accuracy_score(y_train, svm.predict(x_train)))
    return model
    nnmodel = None
    try:
        with open(os.path.join(os.getcwd(), 'GenderNeuralNetParams.json'), 'r') as nnp_file:
            nnmodel = model_from_json(nnp_file.read())
        nnmodel.load_weights(os.path.join(os.getcwd(), 'GenderNeuralNetWeights.h5'))
    except Exception as e:
        nn = Sequential()
        nn.add(Dense(1024, input_dim=2346, activation='sigmoid'))
        nn.add(Dropout(0.1))
        nn.add(Dense(256, activation='sigmoid'))
        nn.add(Dropout(0.05))
        nn.add(Dense(32, activation='sigmoid'))
        # nn.add(Dropout(0.3))
        nn.add(Dense(2, activation='softmax'))
        y_list = y_train.tolist()
        y = np.array(np.zeros((len(y_list), 2)), np.float32)
        for i in range(len(y_list)):
            y[i, int(y_list[i])] = 1
        print(x_train.shape, y.shape)
        print(x_train[0], y[0])
        sgd = SGD(lr=0.005, momentum=0.9)
        nn.compile(loss='mean_squared_error', optimizer=sgd)
        nn.fit(x_train, y, epochs=150, batch_size=1, verbose=2, shuffle=False)
        nnmodel = nn
        params = nnmodel.to_json()
        try:
            with open(os.path.join(os.getcwd(), 'GenderNeuralNetParams.json'), 'w') as nnp_file:
                nnp_file.write(params)
            nnmodel.save_weights(os.path.join(os.getcwd(), 'GenderNeuralNetWeights.h5'))
        except Exception as e:
            print(e)
            pass
    model[1] = nnmodel
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

    model = [None, None]
    return model
    x_train = []
    y_train = []
    try:
        model[0] = load('svm_race.joblib')
    except FileNotFoundError as e:
        knn = SVC(kernel='linear', probability=True)#KNeighborsClassifier(n_neighbors=len(train_image_labels))
        x_train = []
        y_train = []
        for i in range(len(train_image_paths)):
            if train_image_labels[i] == '4':
                print("Nearijevska rasa lol")
                continue
            y_train.append(train_image_labels[i])
            image = load_rgb_image(train_image_paths[i])
            face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
                face_descriptor(image, train_image_paths[i])
            x = []
            x.extend(hue_histogram.flatten().tolist())
            #x.extend(saturation_histogram.flatten().tolist())
            x.extend(value_histogram.flatten().tolist())
            x = np.array(x)
            x_train.append(x.flatten())
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(y_train.shape, x_train.shape)
        x_train, y_train = cc.fit_resample(x_train, y_train)
        print(y_train.shape, x_train.shape)
        print(y_train)
        knn = knn.fit(x_train, y_train)
        model[0] = knn
        dump(knn, 'svm_race.joblib')
        print("Train accuracy: ", accuracy_score(y_train, knn.predict(x_train)))

    nnmodel = None
    try:
        with open(os.path.join(os.getcwd(), 'RaceNeuralNetParams.json'), 'r') as nnp_file:
            nnmodel = model_from_json(nnp_file.read())
        nnmodel.load_weights(os.path.join(os.getcwd(), 'RaceNeuralNetWeights.h5'))
    except Exception as e:
        nn = Sequential()
        nn.add(Dense(256, input_dim=437, activation='sigmoid'))
        nn.add(Dropout(0.1))
        #nn.add(Dense(256, activation='sigmoid'))
        #nn.add(Dropout(0.05))
        nn.add(Dense(64, activation='sigmoid'))
        #nn.add(Dropout(0.3))
        nn.add(Dense(5, activation='softmax'))
        y_list = y_train.tolist()
        y = np.array(np.zeros((len(y_list), 5)), np.float32)
        print(y.shape)
        for i in range(len(y_list)):
            y[i, int(y_list[i])] = 1
        print(x_train.shape, y.shape)
        print(x_train[0], y[0])
        sgd = SGD(lr=0.05, momentum=0.8)
        nn.compile(loss='mean_squared_error', optimizer=sgd)
        nn.fit(x_train, y, epochs=1500, batch_size=1, verbose=2, shuffle=False)
        nnmodel = nn
        params = nnmodel.to_json()
        try:
            with open(os.path.join(os.getcwd(), 'RaceNeuralNetParams.json'), 'w') as nnp_file:
                nnp_file.write(params)
            nnmodel.save_weights(os.path.join(os.getcwd(), 'RaceNeuralNetWeights.h5'))
        except Exception as e:
            print(e)
            pass
    model[1] = nnmodel
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
    return age
    image = load_rgb_image(image_path)
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        face_descriptor(image, image_path)
    temp = trained_model[0].predict(np.array([face_hog.flatten()]))[0]
    print(temp)
    age = temp * 5
    print(age)
    return int(age // 1)
    index = np.argmax(trained_model[1].predict(np.array([face_hog.flatten()])))
    print(index, index*5)
    age = index*5
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
    #return gender
    image = load_rgb_image(image_path)
    face_image, face_landmarks, face_hog, hue_histogram, saturation_histogram, value_histogram = \
        face_descriptor(image, image_path)
    if face_landmarks is None:
        return 2
    #face_landmarks = minimize_landmark_coordinates(face_landmarks)
    face_landmarks = quadratic_relative_landmark_distances(face_landmarks)
    temp = trained_model[0].predict(np.array([face_landmarks.flatten()]))
    print(temp)
    gender = int(temp[0])
    print(gender)
    return 0 if gender <= 0.5 else 1
    index = np.argmax(trained_model[1].predict(np.array([face_landmarks.flatten()])))
    return index


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
    print(len(hue_histogram)+len(saturation_histogram)+len(value_histogram))
    x = []
    x.extend(hue_histogram.flatten().tolist())
    #x.extend(saturation_histogram.flatten().tolist())
    x.extend(value_histogram.flatten().tolist())
    x = np.array(x)
    temp = trained_model[0].predict(np.array([x.flatten()]))
    print(temp)
    race = int(temp[0])
    print(race)
    return race
    index = np.argmax(trained_model[1].predict(np.array([x.flatten()])))
    return index
