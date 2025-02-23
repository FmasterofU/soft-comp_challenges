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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    einvadabingreenimg = cv2.morphologyEx(invadabingreenimg, cv2.MORPH_ERODE, kernel, iterations=1)
    img, contours, hierarchy = cv2.findContours(einvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(einvadabingreenimg, pts=[contour], color=255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    oeinvadabingreenimg = cv2.morphologyEx(einvadabingreenimg, cv2.MORPH_OPEN, kernel, iterations=3)
    _, cellscontours, _ = cv2.findContours(oeinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    red_blood_cell_count = len(cellscontours) - white_blood_cell_count
    has_leukemia = True if red_blood_cell_count/(red_blood_cell_count+white_blood_cell_count) < 0.925 else False
    if white_blood_cell_count > 1 and white_blood_cell_count <= 4:
        m = cv2.moments(whitecellscontours[0])
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        center = [cX, cY]
        canMergeCells = True
        for c in whitecellscontours:
            m = cv2.moments(c)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            if abs(center[0] - cX) < 0.16 * greenimg.shape[1] and abs(center[1] - cY) < 0.16 * greenimg.shape[0]:
                center = [(center[0] + cX) / 2, (center[1] + cY) / 2]
            else:
                canMergeCells = False
                break
        if canMergeCells:
            white_blood_cell_count = 1
            has_leukemia = False
    has_leukemia = True if white_blood_cell_count > 4 and red_blood_cell_count <= 1800 else has_leukemia
    has_leukemia = False if white_blood_cell_count == 1 else has_leukemia
    return red_blood_cell_count, white_blood_cell_count, has_leukemia
