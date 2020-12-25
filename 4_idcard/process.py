import dlib
import cv2
import numpy as np
from PIL import Image
import sys
import os
import pyocr
import pyocr.builders
import re
# import libraries here
import datetime


tools = pyocr.get_available_tools()
tool = tools[0]
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
print("Using backend: %s" % (tool.get_name()))


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None,
                 ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth if date_of_birth is not None else datetime.date.today()
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(models_folder: str, image_path: str) -> Person:
    """
    Procedura prima putanju do foldera sa modelima, u slucaju da su oni neophodni, kao i putanju do slike sa koje
    treba ocitati vrednosti. Svi modeli moraju biti uploadovani u odgovarajuci folder.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param models_folder: <str> Putanja do direktorijuma sa modelima
    :param image_path: <str> Putanja do slike za obradu
    :return:
    """
    person = Person()
    # Person(date_of_birth=datetime.datetime.strptime(row['date_of_birth'], '%Y-%m-%d').date())
    # Person('test', datetime.date.today(), 'test', 'test', 'test')

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    lang = 'eng'
    id_card = extract_id_card(image_path)
    inv_id_cart = np.abs(id_card.astype(np.int16) - np.array([255, 255, 255])).astype(np.uint8)
    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(id_card), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )
    text = ''
    # print('-------------------------------------------------------')
    for i, line in enumerate(line_and_word_boxes):
        # print('line %d' % i)
        # print(line.content, line.position)
        text += line.content + ' '
        # print('boxes')
        # for box in line.word_boxes:
        #     print(box.content, box.position, box.confidence)
        # print()
    # print('-------------------------------------------------------')
    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(inv_id_cart), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )
    for i, line in enumerate(line_and_word_boxes):
        text += line.content + ' '
    dob = parse_dob(text)
    print(dob)
    ssn = parse_ssn(text)
    print(ssn)
    company = get_company(text)
    # print('-------------------------------------------------------')
    return Person(date_of_birth=dob, ssn=ssn, company=company)  # person


def parse_dob(text):
    date_re = re.compile('[0-9]{2}[,] [A-Z][a-z]{2} [0-9]{4}')
    found = date_re.findall(text)
    print(found)
    date = None
    dates = []
    for date_str in found:
        try:
            dates.append(datetime.datetime.strptime(date_str, '%d, %b %Y').date())
        except Exception as e:
            print(e)
    if len(dates) != 0:
        date = min(dates)
    return date


def parse_ssn(text):
    ssn_re_loose = re.compile(r'.{3}[-].{2}[-].{4}')
    found = ssn_re_loose.findall(text)
    print(found)
    ssn_re_tight = re.compile(r'[\d]{3}[-][\d]{2}[-][\d]{4}')
    for ssn_str in found:
        if ssn_re_tight.match(ssn_str) is not None:
            return ssn_str
    if len(found) == 0:
        return None  # '888-88-8888'
    number_re = re.compile(r'[\d]')
    found.sort(key=lambda x: len(number_re.findall(x)), reverse=True)
    return found[0]


def get_company(text):
    if 'IBM' in text:
        return 'IBM'
    elif 'Google' in text:
        return 'Google'
    elif 'Apple' in text:
        return 'Apple'
    else:
        return None  # 'Google'


def extract_id_card(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[int(image.shape[0] * 0.01):int(image.shape[0] * 0.99),
                  int(image.shape[1] * 0.01):int(image.shape[1] * 0.99)]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = 255 - cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 7)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    lines = cv2.HoughLinesP(closing, rho=1, theta=3.14 / 180, threshold=80, minLineLength=100, maxLineGap=10)
    hough_space = np.zeros(closing.shape).astype(np.uint8)
    for line in lines:
        hough_space = cv2.line(hough_space, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color=255, thickness=1)
    points = []
    for line in lines:
        points.append([line[0][0], line[0][1]])
        points.append([line[0][2], line[0][3]])
    hull = cv2.convexHull(np.array(points), False)
    center, size, angle = cv2.minAreaRect(hull)
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = image.shape[0], image.shape[1]
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(image, rot_matrix, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    id_card_rgb = cv2.rotate(img_crop, cv2.ROTATE_90_COUNTERCLOCKWISE) if angle <= -45 else img_crop
    return id_card_rgb
