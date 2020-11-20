import cv2
import collections
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.models import model_from_json
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import scipy.signal
import os
from fuzzywuzzy import fuzz


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
    if train_image_paths[0][-5] == '1':
        train_image_paths = train_image_paths[::-1]
    nn = Sequential()
    nn.add(Dense(256, input_dim=1024, activation='sigmoid'))
    nn.add(Dropout(0.3))
    #nn.add(Dense(256, activation='sigmoid'))
    #nn.add(Dropout(0.3))
    nn.add(Dense(128, activation='sigmoid'))
    nn.add(Dropout(0.3))
    nn.add(Dense(len(alphabet), activation='softmax'))
    y = np.array(np.eye(len(alphabet)), np.float32)
    x = []
    for path in train_image_paths:
        vectorimagerois, _ = extract_rois(path)
        x.extend(vectorimagerois)
    x = np.array(x)
    sgd = SGD(lr=0.03, momentum=0.9)
    nn.compile(loss='mean_squared_error', optimizer=sgd)
    nn.fit(x, y, epochs=4000, batch_size=1, verbose=2, shuffle=False)
    return nn


def nn_predict_text(trained_model, vectorcharimgrois):
    extracted_text = ''
    for i in range(len(vectorcharimgrois)):
        if vectorcharimgrois[i].ndim == 1:
            vectorcharimgrois[i] = np.array([vectorcharimgrois[i]])
        index = np.argmax(trained_model.predict(vectorcharimgrois[i]))
        extracted_text += alphabet[index]
    return extracted_text


def add_spaces_to_nn_text_output(extracted_text, distancerois):
    distances = np.array(distancerois).reshape(len(distancerois), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(distances)
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    charsnum = len(extracted_text)
    for i in range(charsnum):
        if i < len(distancerois) and k_means.labels_[i] == w_space_group:
            extracted_text = extracted_text[:i+1] + ' ' + extracted_text[i+1:]
    return extracted_text


def guess_text_by_distance(extracted_text, vocabulary):
    words = extracted_text.split(' ')
    extracted_text = ''
    for i in range(0, len(words)):
        wordguess = []
        for vword in vocabulary.keys():
            wordguess.append((vword, fuzz.ratio(words[i], vword), vocabulary[vword]))
        wordguess.sort(key=lambda x: x[1], reverse=True)
        for j in range(0, len(wordguess)):
            if j == 0:
                continue
            elif wordguess[0][1] != wordguess[j][1]:
                wordguess = wordguess[:j]
                break
        wordguess.sort(key=lambda x: x[2], reverse=True)
        extracted_text += wordguess[0][0]
        if i + 1 != len(words):
            extracted_text += ' '
    return extracted_text


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
    vectorcharimgrois, distancerois = extract_rois(image_path)
    if vectorcharimgrois is None: return extracted_text
    extracted_text = nn_predict_text(trained_model, vectorcharimgrois)
    print("NeuralNet, preprocessed, predicted characters :  ", extracted_text)
    extracted_text = add_spaces_to_nn_text_output(extracted_text, distancerois)
    print("Kmeans, added spaces after aneuralnet results:   ", extracted_text)
    extracted_text = guess_text_by_distance(extracted_text, vocabulary)
    print("Levenshtein, guess word by distance, end result: ", extracted_text)
    return extracted_text


def histogram(image, xmax):
    height, width = image.shape[0:2]
    x = range(0, xmax+1)
    y = np.zeros(xmax+1)
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    return x, y


def distinctHist(image, xmax, sourcevalid):
    height, width = image.shape[0:2]
    x = range(0, xmax+1)
    y = np.zeros(xmax+1)
    for i in range(0, height):
        for j in range(0, width):
            if sourcevalid[i, j]:
                pixel = image[i, j]
                y[pixel] += 1
    return x, y


def rectPoints(r):
    pts = [[r[0],r[1]], [r[0],r[1]], [r[0],r[1]], [r[0],r[1]]]
    pts[1][0] = pts[1][0] + r[2]
    pts[2][0] = pts[1][0]
    pts[2][1] = pts[2][1] + r[3]
    pts[3][1] = pts[2][1]
    return pts


def isInside(rectangle, contour):
    pts = rectPoints(rectangle)
    rectcontour = cv2.convexHull(np.array([pts[0], pts[1], pts[2], pts[3]], dtype=np.int32))
    for coor in contour:
        point = (coor[0][0], coor[0][1])
        if cv2.pointPolygonTest(rectcontour, point, False) < 0:
            return False
    return True


def expandRect(rectangle):
    wxsideshift = 0#int(0.15 * rectangle[2])
    hyupshift = int(0.5 * rectangle[3])
    hydownshift = int(0.15 * rectangle[3])
    rectw = int(2 * wxsideshift + rectangle[2])
    recth = int(hyupshift + hydownshift + rectangle[3])
    rectx = rectangle[0] - wxsideshift
    recty = rectangle[1] - hyupshift
    return (rectx, recty, rectw, recth)


def cropMultipleContoursBindingRect(baseimg, cnts):
    img = np.copy(baseimg)
    y1, x1 = img.shape
    x2 = 0
    y2 = 0
    for cnt in cnts:
        #cv2.drawContours(img, [cnt], -1, 255)
        x, y, w, h = cv2.boundingRect(cnt)
        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x2 < x + w:
            x2 = x + w
        if y2 < y + h:
            y2 = y + h
    w = x2 - x1
    h = y2 - y1
    rect = (x1, y1, w, h)
    return [rect , baseimg[int(y1):int(y1+h+1), int(x1):int(x1+w+1)]]


def rectDistance(r1, r2):
    ret = r2[0] - r1[0] - r2[2]
    return ret if ret >= 0 else 0


def extract_rois(image_path):
    cvimg = cv2.imread(image_path)
    rgbimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    hsvImage = cv2.cvtColor(cvimg, cv2.COLOR_BGR2HSV)
    x, y = histogram(hsvImage[:, :, 0], 179)
    pixels = hsvImage.shape[0] * hsvImage.shape[1]
    for i in range(len(y)):
        if not 0.05 * pixels < y[i] < 0.3 * pixels:
            y[i] = 0
    peaks, _ = scipy.signal.find_peaks(y)
    peakcandidate = []
    for peak in peaks:
        valid = True
        hsvpeak = [peak]
        for i in range(1, 3):
            xtemp, ytemp = distinctHist(hsvImage[:, :, i], 255, hsvImage[:, :, 0] == peak)
            hsvpeak.append(np.argmax(ytemp))
            if (i==1 and np.amax(ytemp)<0.3*y[peak]):# or (i==2 and np.amax(ytemp)<0.4*y[peak]):#if (i == 1 and np.amax(ytemp) < 0.6 * y[peak]) or (i == 2 and np.amax(ytemp) < 0.4 * y[peak]):
                valid = False
                break
        if valid:
            peakcandidate.append(hsvpeak)
    peakcandidate.sort(key=lambda x: x[2], reverse=True)
    textimg = np.zeros(hsvImage[:,:,0].shape, hsvImage[:,:,0].dtype)
    print(len(peakcandidate), image_path, peakcandidate)
    if len(peakcandidate) == 0:
        return None, None
    textimg[np.logical_and(hsvImage[:, :, 0] == peakcandidate[0][0], hsvImage[:, :, 1] == peakcandidate[0][1])] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedopentextimg = textimg
    # opentextimg = cv2.morphologyEx(textimg,cv2.MORPH_OPEN,kernel, iterations = 1)
    # closedopentextimg = cv2.morphologyEx(opentextimg,cv2.MORPH_CLOSE,kernel, iterations = 2)
    img, contours, hierarchy = cv2.findContours(closedopentextimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) <= 1:
            contours = contours[:i]
            break
    exrects = []
    i = 0
    while i < len(contours):
        j = i + 1
        exrect = [expandRect(cv2.boundingRect(contours[i])), []]
        while j < len(contours):
            if isInside(exrect[0], contours[j]):
                exrect[1].append(contours[j])
                contours.pop(j)
                continue
            j = j + 1
        exrects.append(exrect)
        i = i + 1
    exrects.sort(key=lambda x: x[0][2] * x[0][3], reverse=True)
    for i in range(0, len(exrects)):
        if exrects[i][0][2] * exrects[i][0][3] < 0.1 * exrects[0][0][2] * exrects[0][0][3]:
            exrects = exrects[:i]
            break
    rois = []
    #baseimg = np.zeros(textimg.shape)
    for rect in exrects:
        char = cropMultipleContoursBindingRect(closedopentextimg, rect[1])
        #char = [rect[0], closedopentextimg[rect[0][1]:rect[0][1]+rect[0][3]+1, rect[0][0]:rect[0][0]+rect[0][2]+1]]
        rois.append(char)
    rois.sort(key=lambda x: x[0][0])
    vectorimgrois = []
    distancerois = []
    print(len(exrects))
    for i in range(0, len(rois)):
        if rois[i][1].shape[0]==0 or rois[i][1].shape[1]==0: continue
        vectorimgrois.append(cv2.resize(rois[i][1]/255, (32, 32), interpolation=cv2.INTER_NEAREST).flatten())
        if i + 1 < len(rois):
            distancerois.append(rectDistance(rois[i][0], rois[i + 1][0]))
    return vectorimgrois, distancerois
