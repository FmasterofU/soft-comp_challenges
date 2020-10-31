# import libraries here
import numpy as np
import cv2


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0
    has_leukemia = None

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    cvimg = cv2.imread(image_path)
    rgbimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    gsimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    greenimg = cvimg[:, :, 1].astype('float64')
    greenimg *= (255.0 / greenimg.max())
    greenimg = greenimg.astype('uint8')
    _, bingreenimg = cv2.threshold(greenimg, 127, 255, cv2.THRESH_BINARY)
    invbingreenimg = 255 - bingreenimg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erodedinvbingreenimg = cv2.erode(invbingreenimg, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilatederodedinvbingreenimg = cv2.dilate(erodedinvbingreenimg, kernel, iterations=5)
    img, contours, hierarchy = cv2.findContours(dilatederodedinvbingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    white_blood_cell_count = len(contours)
    _, bingsimg = cv2.threshold(gsimg, 192, 255, cv2.THRESH_BINARY)
    invbingsimg = 255 - bingsimg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    erodedinvbingsimg = cv2.erode(invbingsimg, kernel, iterations=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilatederodedinvbingsimg = cv2.dilate(erodedinvbingsimg, kernel, iterations=4)
    img, contours, hierarchy = cv2.findContours(dilatederodedinvbingsimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    red_blood_cell_count = len(contours)
    has_leukemia = True if red_blood_cell_count/(red_blood_cell_count+white_blood_cell_count)<0.925 else False
    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    return red_blood_cell_count, white_blood_cell_count, has_leukemia
