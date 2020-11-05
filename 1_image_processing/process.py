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
    _, whitecellscontours, _ = cv2.findContours(oinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    white_blood_cell_count = len(whitecellscontours)
    adabingreenimg = cv2.adaptiveThreshold(greenimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 535, 0)
    invadabingreenimg = 255 - adabingreenimg
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erodedinvadabingreenimg = cv2.morphologyEx(invadabingreenimg, cv2.MORPH_ERODE, kernel, iterations=2)
    img, contours, hierarchy = cv2.findContours(erodedinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(erodedinvadabingreenimg, pts=[contour], color=255)
    _, contours, _ = cv2.findContours(erodedinvadabingreenimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    redcellcount = 0
    for contour in contours:
        cimg = np.zeros(shape=erodedinvadabingreenimg.shape, dtype=np.uint8)
        cv2.fillPoly(cimg, pts=[contour], color=255)
        whites = 0
        for wcontour in whitecellscontours:
            wimg = np.zeros(shape=erodedinvadabingreenimg.shape, dtype=np.uint8)
            cv2.fillPoly(wimg, pts=[wcontour], color=255)
            if cv2.countNonZero(cv2.bitwise_and(cimg, wimg)) != 0:
                whites = whites + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                wimg = cv2.morphologyEx(wimg, cv2.MORPH_DILATE, kernel, iterations=2)
                cimg = cv2.subtract(cimg, wimg)
        if whites == 0:
            redcellcount = redcellcount + 1
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cimg = cv2.morphologyEx(cimg, cv2.MORPH_OPEN, kernel, iterations=3)
            _, cimgcontours, _ = cv2.findContours(cimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            redcellcount = redcellcount + len(cimgcontours)
    red_blood_cell_count = redcellcount
    has_leukemia = True if red_blood_cell_count/(red_blood_cell_count+white_blood_cell_count)<0.925 else False
    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure
    #if white_blood_cell_count<=2 and red_blood_cell_count/white_blood_cell_count>12:
    #    has_leukemia = False
    return red_blood_cell_count, white_blood_cell_count, has_leukemia
