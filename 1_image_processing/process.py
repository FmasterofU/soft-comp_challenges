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
    gsimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    greenimg = cvimg[:, :, 1].astype('float64')
    greenimg *= (255.0 / greenimg.max())
    greenimg = greenimg.astype('uint8')
    adabingreenimg = cv2.adaptiveThreshold(greenimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 535, 62)
    invadabingreenimg = 255 - adabingreenimg
    img, contours, hierarchy = cv2.findContours(invadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(invadabingreenimg, pts=[contour], color=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    oinvadabingreenimg = cv2.morphologyEx(invadabingreenimg, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coinvadabingreenimg = cv2.morphologyEx(oinvadabingreenimg, cv2.MORPH_CLOSE, kernel, iterations=2)
    dtcoinvadabingreenimg = cv2.distanceTransform(coinvadabingreenimg, cv2.DIST_L2, 5)
    _, temp = cv2.threshold(dtcoinvadabingreenimg, 0.7 * dtcoinvadabingreenimg.max(), 255, 0)
    defwhitecells = np.uint8(temp)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    whitecells = cv2.dilate(defwhitecells, kernel, iterations=1)
    _, whitecellscontours, _ = cv2.findContours(whitecells, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    white_blood_cell_count = len(whitecellscontours)
    adabingreenimg = cv2.adaptiveThreshold(greenimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 535, 0)
    invadabingreenimg = 255 - adabingreenimg
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    surewhitecells = cv2.dilate(coinvadabingreenimg, kernel, iterations=3)
    redinvadabingreenimg = cv2.subtract(invadabingreenimg, surewhitecells)
    img, contours, hierarchy = cv2.findContours(redinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(redinvadabingreenimg, pts=[contour], color=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erodedredinvadabingreenimg = cv2.erode(redinvadabingreenimg, kernel, iterations=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilatederodedredinvadabingreenimg = cv2.dilate(erodedredinvadabingreenimg, kernel, iterations=2)
    dtdilatederodedredinvadabingreenimg = cv2.distanceTransform(dilatederodedredinvadabingreenimg, cv2.DIST_L2, 5)
    _, temp = cv2.threshold(dtdilatederodedredinvadabingreenimg, 0.35 * dtdilatederodedredinvadabingreenimg.max(), 255, 0)
    defredcells = np.uint8(temp)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    redcells = cv2.dilate(defredcells, kernel, iterations=1)
    _, redcellscontours, _ = cv2.findContours(redcells, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    red_blood_cell_count = len(redcellscontours)
    has_leukemia = True if red_blood_cell_count/(red_blood_cell_count+white_blood_cell_count)<0.925 else False
    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure
    #if white_blood_cell_count<=2 and red_blood_cell_count/white_blood_cell_count>12:
    #    has_leukemia = False
    return red_blood_cell_count, white_blood_cell_count, has_leukemia
