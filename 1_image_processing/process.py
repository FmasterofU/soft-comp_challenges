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
    greenimg = cvimg[:, :, 1].astype('float64')
    greenimg *= (255.0 / greenimg.max())
    greenimg = greenimg.astype('uint8')
    adabingreenimg = cv2.adaptiveThreshold(greenimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 535, 62)
    invadabingreenimg = 255 - adabingreenimg
    img, contours, hierarchy = cv2.findContours(invadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(invadabingreenimg, pts=[contour], color=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    oinvadabingreenimg = cv2.morphologyEx(invadabingreenimg, cv2.MORPH_OPEN, kernel, iterations=3)
    _, whitecellscontours, _ = cv2.findContours(oinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    white_blood_cell_count = len(whitecellscontours)
    adabingreenimg = cv2.adaptiveThreshold(greenimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 535, 0)
    invadabingreenimg = 255 - adabingreenimg
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #einvadabingreenimg = cv2.morphologyEx(invadabingreenimg, cv2.MORPH_ERODE, kernel, iterations=1)
    einvadabingreenimg = invadabingreenimg
    img, contours, hierarchy = cv2.findContours(einvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(einvadabingreenimg, pts=[contour], color=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    oeinvadabingreenimg = cv2.morphologyEx(einvadabingreenimg, cv2.MORPH_OPEN, kernel, iterations=3)
    _, cellscontours, _ = cv2.findContours(oeinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contourmin = greenimg.shape[0] * greenimg.shape[1]
    contourmax = 0
    for c in cellscontours:
        cimg = np.zeros(shape=greenimg.shape, dtype=np.uint8)
        cv2.fillPoly(cimg, pts=[c], color=255)
        cflat = c.flatten()
        temp = cv2.countNonZero(cimg)
        if 0 not in cflat or cimg.shape[0] - 1 not in cflat or cimg.shape[1] - 1 not in cflat:
            if contourmin > temp:
                contourmin = temp
        if contourmax < temp:
            contourmax = temp
    meancontourarea = (contourmax + contourmin) / 2
    cellsnum = 0
    for c in cellscontours:
        cflat = c.flatten()
        if 0 not in cflat or greenimg.shape[0] - 1 not in cflat or greenimg.shape[1] - 1 not in cflat:
            if cv2.contourArea(c) <= 0.05 * meancontourarea:
                continue
        cellsnum = cellsnum + 1
    red_blood_cell_count = cellsnum - white_blood_cell_count
    has_leukemia = True if red_blood_cell_count/(red_blood_cell_count+white_blood_cell_count)<0.925 else False
    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure
    #if white_blood_cell_count<=2 and red_blood_cell_count/white_blood_cell_count>12:
    #    has_leukemia = False
    return red_blood_cell_count, white_blood_cell_count, has_leukemia
