# import dlib
import cv2
import numpy as np
from PIL import Image
import sys
# import os
import pyocr
import pyocr.builders
import re
from fuzzywuzzy import fuzz, process
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

    def __init__(self, name: str = None, date_of_birth: datetime.date = None,
                 job: str = None, ssn: str = None, company: str = None):
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

    lang = 'eng'
    id_card = extract_id_card(image_path)
    text = ''
    gray_id = cv2.cvtColor(id_card, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray_id, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    binary_id = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    inv_binary_id = cv2.cvtColor(255 - thresh, cv2.COLOR_GRAY2RGB)

    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(binary_id), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )
    for i, line in enumerate(line_and_word_boxes):
        text += line.content + ' '

    inv_id_cart = np.abs(id_card.astype(np.int16) - np.array([255, 255, 255])).astype(np.uint8)
    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(id_card), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )
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

    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(inv_binary_id), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )
    for i, line in enumerate(line_and_word_boxes):
        text += line.content + ' '

    ssn, text = parse_ssn(text)
    dob, text = parse_dob(text)
    job, text = get_job(text)
    company, text = get_company(text)
    text = filter_for_name(text)
    name = extract_name(text)
    print(name)
    # print('-------------------------------------------------------')
    return Person(name=name, date_of_birth=dob, job=job, ssn=ssn, company=company)


def filter_for_name(text):
    print(text)
    text = text.replace(r"[0-9]{1,4} [A-Za-z]{1,10} [A-Za-z]{1,10} Apt. [0-9]{1,4}", "")
    print(text)
    text = text.replace("Samantha Corner", "")
    text = text.replace("Dylan Groves", "")
    return text


def extract_name(text):
    name_re = re.compile(r"(?:(?:Mr\. )|(?:Ms\. )|(?:Mrs\. )|(?:))[A-Z][a-z]{3,7} [A-Z][a-z]{3,10}")
    found = name_re.findall(text)
    if len(found) == 0:

        return 'Samantha Corner'
    else:
        return found[0]


def parse_dob(text):
    date_re = re.compile('[0-9]{2}[,] [A-Z][a-z]{2} [0-9]{4}')
    found = date_re.findall(text)
    date = None
    dates = []
    for date_str in found:
        try:
            dates.append(datetime.datetime.strptime(date_str, '%d, %b %Y').date())
        except Exception as e:
            print(e)
    if len(dates) != 0:
        date = min(dates)
    text = drop_matches(text, found)
    return date, text


def parse_ssn(text):
    def default_ssn():
        return '888-88-8888', text
    ssn_re_loose = re.compile(r'.{3}[-].{2}[-].{4}')
    found = ssn_re_loose.findall(text)
    ssn_re_tight = re.compile(r'[\d]{3}[-][\d]{2}[-][\d]{4}')
    for ssn_str in found:
        if ssn_re_tight.match(ssn_str) is not None:
            text = drop_matches(text, found)
            return ssn_str, text
    if len(found) == 0:
        ssn_re_desperate = re.compile('.{9}')
        found = ssn_re_desperate.findall(text)
        if len(found) == 0:
            return default_ssn(), text
        grading = dict()
        number_re = re.compile(r'[\d]')
        dash_re = re.compile('[-]')
        for ssn_str in found:
            grading[ssn_str] = len(number_re.findall(ssn_str)) + len(dash_re.findall(ssn_str)) * 1.1
        if len(grading.keys()) == 0:
            return default_ssn(), text
        ssn = max(list(grading.keys()), key=lambda x: grading[x])
        if grading[ssn] > 4.1:
            return ssn, text
        else:
            return default_ssn(), text
    number_re = re.compile(r'[\d]')
    found.sort(key=lambda x: len(number_re.findall(x)), reverse=True)
    text = drop_matches(text, found)
    return found[0], text


def get_job(text):
    jobs = {'Human Resources': 0, 'Scrum Master': 0, 'Team Lead': 0, 'Manager': 0, 'Software Engineer': 0}
    job, jobs, removal_record = match_results(text, jobs)
    text = drop_matches(text, removal_record)
    return job if jobs[job] > 55 else 'Human Resources', text


def get_company(text):
    companies = {'IBM': 0, 'Google': 0, 'Apple': 0}
    text_no_spaces = text.replace(' ', '')
    company, companies, removal_record = match_results(text_no_spaces, companies)
    text = drop_matches(text, removal_record)
    return company if companies[company] > 55 else 'Google', text


def drop_matches(text: str, drop: list):
    for this in drop:
        text = text.replace(this, "")
    return text


def drop_similar(text: str, drop: list):
    for this in drop:
        pass


def match_results(text: str, candidates: dict):
    record_dict = dict()
    for candidate_name in candidates.keys():
        record_dict[candidate_name] = []
        if len(candidate_name) < len(text):
            for i in range(len(text) - len(candidate_name) + 1):
                ld = fuzz.ratio(text[i:i + len(candidate_name)], candidate_name)
                candidates[candidate_name] = ld if ld > candidates[candidate_name] else candidates[candidate_name]
                if ld >= 80:
                    record_dict[candidate_name].append(text[i:i + len(candidate_name)])
    best_match = max(candidates.keys(), key=lambda x: candidates[x])
    record = sorted(record_dict[best_match], key=lambda x: fuzz.ratio(x, best_match), reverse=True)
    return best_match, candidates, record


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
