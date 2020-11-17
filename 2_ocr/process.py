import cv2
import collections
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import os


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """

    try:
        with open(os.path.join(serialization_folder, 'NeuralNetParams.json'), 'r') as nnp_file:
            nnmodel = model_from_json(nnp_file.read())
        nnmodel.load_weights(os.path.join(serialization_folder, 'NeuralNetWeights.h5'))
    except Exception as e:
        nnmodel = train_ocr(train_image_paths)
        params = nnmodel.to_json()
        try:
            with open(os.path.join(serialization_folder, 'NeuralNetParams.json'), 'w') as nnp_file:
                nnp_file.write(params)
            nnmodel.save_weights(os.path.join(serialization_folder, 'NeuralNetWeights.h5'))
        except Exception as e:
            print(e)
            pass
    return nnmodel


alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']


def train_ocr(train_image_paths):
    nn = Sequential()
    nn.add(Dense(256, input_dim=1024, activation='sigmoid'))
    nn.add(Dense(len(alphabet)))
    y = np.array(np.eye(len(alphabet)), np.float32)
    for path in train_image_paths:
        pass
    x = None
    sgd = SGD(lr=0.01, momentum=0.9)
    nn.compile(loss='mean_squared_error', optimizer=sgd)
    nn.fit(x, y, epochs=500, batch_size=1, verbose=0, shuffle=False)
    return nn


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    return extracted_text


