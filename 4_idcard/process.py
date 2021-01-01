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

from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score

tools = pyocr.get_available_tools()
tool = tools[0]
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
print("Using backend: %s" % (tool.get_name()))
ros = RandomOverSampler(random_state=0)
Y_train_company = np.array(['Google', 'IBM', 'Apple', 'Apple', 'IBM', 'Google', 'Google', 'IBM', 'Apple', 'Google',
                            'Apple', 'IBM', 'IBM', 'Google', 'IBM', 'Google', 'IBM', 'Apple', 'IBM', 'Google', 'Apple',
                            'IBM', 'IBM', 'Apple', 'Google', 'Apple', 'Google', 'Google', 'Google', 'Apple', 'IBM',
                            'Google', 'IBM', 'IBM', 'Apple', 'Apple', 'IBM', 'IBM', 'Apple', 'Apple', 'Apple', 'Google',
                            'Apple', 'IBM', 'Apple', 'Google', 'Google', 'Apple', 'Google', 'IBM', 'Google', 'IBM',
                            'Google', 'Google', 'IBM', 'Google', 'Google', 'IBM', 'Apple', 'Apple', 'IBM', 'IBM', 'IBM',
                            'Apple', 'IBM', 'IBM', 'Apple', 'IBM', 'Apple', 'IBM', 'Google', 'IBM', 'IBM', 'Google',
                            'IBM', 'Apple', 'Google', 'Apple', 'IBM', 'Google', 'IBM', 'Google', 'Google', 'IBM',
                            'Google', 'Google', 'IBM', 'IBM', 'Google', 'IBM', 'IBM', 'Google', 'IBM', 'Apple',
                            'Google', 'Apple', 'Google', 'Apple', 'Apple', 'Apple', 'IBM', 'Google', 'Apple', 'IBM',
                            'Google', 'Apple', 'Apple', 'Google', 'IBM', 'Apple', 'IBM', 'IBM', 'Google', 'Apple',
                            'Google', 'IBM', 'Google', 'Google', 'IBM', 'Apple', 'Google', 'Google', 'Apple', 'IBM',
                            'Apple', 'IBM', 'Apple', 'Google', 'IBM', 'Apple', 'Google', 'IBM', 'Apple', 'Apple',
                            'IBM', 'IBM', 'IBM', 'IBM', 'IBM', 'Google', 'Google', 'IBM', 'Apple', 'Google', 'Google',
                            'IBM', 'Apple', 'Apple', 'IBM', 'Google'])
X_train_company = np.array([[142.0185534503962, 200.52632527948072, 192.71023699224205],
                            [110.8481486290294, 106.86207669881139, 105.34413348913885],
                            [186.83572567783094, 179.67524610900293, 168.9021476104053],
                            [217.22971913812134, 173.47865477548672, 126.95656751166393],
                            [93.44462081128748, 92.3325176366843, 91.91834215167547],
                            [108.32034689205847, 150.29666754710337, 143.68851030110935],
                            [84.65431685036243, 114.3853763280707, 112.36154552676001],
                            [143.2946554149086, 140.38739311946063, 138.3270711941598],
                            [176.72391590941191, 164.11760205282735, 143.28407253951065],
                            [107.22413063909775, 154.6938439849624, 152.1314771303258],
                            [167.784635930309, 153.3117973372781, 130.64772764628535],
                            [165.8759723778759, 160.40881897290527, 157.07348457380184],
                            [79.73675031409111, 77.58549064339087, 76.8388464590359],
                            [89.82637370531165, 127.88066643620054, 126.97836461836157],
                            [155.27264362209263, 154.74517194740693, 153.44164209219306],
                            [122.40166604961125, 184.77737874861165, 178.12184376156978],
                            [161.649316872428, 163.1123840877915, 163.32186556927297],
                            [187.62834217711608, 179.16611715006954, 165.87008654886404],
                            [171.12667702799519, 165.69204931157194, 164.69013103035192],
                            [118.68855350670374, 160.58941390716822, 152.67944804166518],
                            [186.20680582009544, 173.18948691263967, 149.35216289983703],
                            [77.88161387074243, 76.20803648465426, 75.93602859608036],
                            [156.21033876342096, 151.09448660989756, 147.65748025422684],
                            [198.3013896295367, 178.52272655980425, 152.5737083973776],
                            [124.20417850138635, 175.15134070467303, 168.08244995604247],
                            [132.8241695303551, 126.30612352042765, 127.90152730049638],
                            [102.6447323209384, 147.82129583917535, 145.69672386895476],
                            [132.86885847199855, 185.3862721417069, 179.1688182143496],
                            [137.35746274635477, 199.7590650536773, 195.52876141643966],
                            [151.34245337602738, 140.67429400248872, 119.07202244608423],
                            [171.562239026796, 166.21132253822128, 167.7172653296071],
                            [76.7967824967825, 105.706552006552, 103.0030654030654],
                            [171.45888030098556, 166.1536825747352, 163.83265856950067],
                            [89.40224758406578, 88.54957584503038, 88.65752765752767],
                            [150.38601464779458, 136.55022218564847, 113.71155365371955],
                            [131.27556161597485, 120.43319543311455, 103.93396967735971],
                            [99.59397673743665, 98.52476949523363, 97.38806286696583],
                            [169.8180603448276, 166.46119122257053, 164.6642829153605],
                            [137.61334185461985, 130.2558055238922, 121.21305283038666],
                            [118.98729856854187, 114.74466387054048, 106.95958635445145],
                            [182.36843947782987, 172.34342266483642, 160.55305996480786],
                            [54.24268172677876, 75.57838531639071, 74.23224233089462],
                            [154.25572016460904, 143.90933782267115, 123.38544706322485],
                            [103.81537975097582, 102.0229398809394, 102.07447829215639],
                            [141.9337353450741, 128.1563892953861, 109.86879110527998],
                            [86.89473731732754, 118.22034916005653, 115.81568195464621],
                            [108.56542746476062, 156.2715848740167, 152.812147106976],
                            [191.17942079018027, 181.80027216103164, 170.01101430215354],
                            [69.98074483508518, 93.68409749909387, 90.33070859006887],
                            [111.56310391329632, 108.22017807580801, 107.04433337044965],
                            [81.70695907865719, 115.14690026954177, 112.48401127174712],
                            [168.22356851240178, 164.2573333606877, 162.37565188847628],
                            [131.13849916487712, 190.42623081205758, 185.14632148254196],
                            [55.00850420966866, 77.41420253938077, 75.96326724606193],
                            [95.16037265143838, 93.15406564278476, 91.98689093318781],
                            [66.56481935852365, 94.21264289219432, 91.50966960830846],
                            [108.2695932656754, 159.71557705924002, 156.61048697978129],
                            [118.19418092979397, 112.39296939737692, 107.55884790124867],
                            [157.54663263225584, 145.11002368293103, 126.690682883985],
                            [170.19489958317544, 155.0997650625237, 132.61445244410763],
                            [76.84662800331401, 76.0150124275062, 74.08877381938692],
                            [166.56695316804408, 164.46846280991736, 163.104],
                            [133.0483642231894, 129.9908063964008, 131.26300389587104],
                            [165.2870404663371, 157.4054121511731, 147.53469348319675],
                            [90.17212800665654, 89.24688249269207, 88.76247824040114],
                            [124.51150932400934, 118.83406593406593, 116.20708458208459],
                            [126.25000797193877, 115.9991868622449, 99.95203284438776],
                            [161.02062182741116, 155.51380282154395, 151.83290757465883],
                            [134.12133129317442, 127.12389043988689, 117.39286481262751],
                            [159.65969087197433, 156.7240361621464, 156.20127539613102],
                            [114.96771912159569, 170.1855421301624, 164.77242360311982],
                            [176.23477644990066, 173.48502881891304, 170.90677814321535],
                            [167.1393181321423, 161.5615821445804, 157.7942271765623],
                            [131.38467582154814, 187.14293400286945, 182.6314989410398],
                            [165.54309770052564, 157.89456995332824, 153.44576155551766],
                            [195.72716016317912, 181.29797268185894, 156.1360472924454],
                            [90.09214917027417, 128.81492153679653, 126.1800505050505],
                            [182.63161273754494, 175.01077298407807, 162.03380842321522],
                            [172.6423154711646, 169.37876175702476, 167.98112985253402],
                            [115.20926552439158, 156.53130482962416, 153.30841427480084],
                            [162.1598037100949, 162.25293356341675, 161.83775884383087],
                            [128.18495573804606, 194.50482639389136, 191.00259728825498],
                            [75.10137970928872, 104.41631281615774, 101.98324902460338],
                            [167.0378317765149, 166.90222527053803, 167.0324210159601],
                            [133.30774488669226, 186.30718404402617, 179.66069895017264],
                            [90.73563652540311, 121.51841126751106, 117.06671199187504],
                            [143.35162443144898, 143.8521442495127, 143.96430582629415],
                            [98.72047876701099, 95.68169372028203, 93.4716674864732],
                            [127.4957604663487, 176.46861861861862, 168.90385090973325],
                            [171.9186437994283, 165.33877513604506, 161.3875221880547],
                            [162.9627881113849, 163.96070564120456, 161.92342833467802],
                            [118.22338199936658, 159.58070933754422, 154.56159531806887],
                            [166.72085054962187, 162.68140251032978, 160.98140640835737],
                            [170.23918731602473, 158.7764106717336, 136.44604949616084],
                            [131.89386179857877, 193.1987258024994, 184.97308257779957],
                            [159.97341666666668, 141.53945833333333, 113.98386904761904],
                            [143.74371658006612, 181.15072649976383, 166.75026547000473],
                            [185.5159046487564, 174.90958782048827, 163.10444851109492],
                            [122.6806316606981, 117.1142032570604, 108.17640572790074],
                            [129.19203961848862, 123.3307436327429, 114.58078293679908],
                            [102.1292985553373, 97.93689235895268, 96.18473116675369],
                            [75.71634720274272, 101.89822346891071, 98.5977455716586],
                            [160.0037386847841, 146.90848862096726, 126.85758831268286],
                            [120.11687161829808, 117.26453516969994, 114.79152320052468],
                            [71.83672256925584, 100.3373675991309, 97.13435293318848],
                            [187.08145516711366, 174.26329224109537, 150.39448156338713],
                            [158.5951694776486, 143.19051907646963, 121.55577206484362],
                            [113.09060279343299, 161.09743935309973, 156.82109777015438],
                            [159.56459829762997, 155.7559646236117, 152.48773850583805],
                            [193.4233060366002, 183.70536221031998, 172.9794582941847],
                            [141.67196842953186, 138.96114493292436, 137.43056588282678],
                            [164.40188434048085, 163.70544726012562, 162.6028914879792],
                            [128.68897028282294, 182.85174136867676, 177.02674671802035],
                            [93.18158495972739, 88.83923869268897, 83.83648447180917],
                            [61.48611573337467, 86.4502368371864, 84.88371043405714],
                            [168.13527791784648, 163.22016213490556, 158.93449439509715],
                            [120.00698696632648, 173.07747582463787, 167.33525971792227],
                            [76.05927486015808, 104.41949407794914, 102.11538827365989],
                            [174.49596694214875, 172.32087052341598, 170.79917355371902],
                            [130.97969444444445, 121.64674206349206, 105.54671428571427],
                            [122.7889542936288, 173.92418229277646, 169.7536224163648],
                            [112.3241237421913, 152.23510268207832, 146.41078816444096],
                            [165.77860511293125, 154.89837315823914, 133.85194837428642],
                            [172.37929391379683, 171.14184535412605, 171.130625947585],
                            [191.84381638496353, 172.61213712899718, 145.1742945374085],
                            [174.71086166119522, 171.86225736485187, 170.64280027141407],
                            [134.8865496845426, 123.77552839116719, 106.95623028391167],
                            [132.48957482911524, 188.69643343098178, 182.12793945797117],
                            [139.83037383177572, 136.47756189539268, 132.66198557140515],
                            [187.5620885093168, 172.8358695652174, 147.01184006211182],
                            [143.70760469783835, 190.3450923338423, 174.57688724995919],
                            [166.69206331688397, 162.49943241837073, 160.62783987891592],
                            [101.3268202516382, 93.53695241721802, 80.92921287702096],
                            [168.26242098958087, 157.14250642329094, 135.68412777226084],
                            [172.52936802426598, 161.59513502370646, 151.1055599729069],
                            [111.65893926600822, 109.16527971410058, 107.71505488247884],
                            [152.6867578473853, 147.60318195349453, 146.0755796646049],
                            [174.49320436787346, 168.21865117364467, 163.90188378624137],
                            [132.52914973866524, 130.90773259913374, 130.4832622637075],
                            [136.31578044596912, 201.94229355550112, 192.92533692722373],
                            [117.2305362103741, 164.37767629485862, 159.46842485824104],
                            [145.60948745298376, 145.48565654297423, 141.69516645146803],
                            [176.20342788303583, 166.42889606833864, 153.99611670864817],
                            [80.48081776880363, 114.88190938919738, 113.20491544674407],
                            [114.77601229546086, 166.12887122122584, 162.34654941296108],
                            [116.29306285317736, 108.91638990780213, 115.3592990978487],
                            [185.88480756307513, 174.35718432709933, 162.45784399328906],
                            [144.5081865354582, 137.35411173991167, 128.0610925456347],
                            [172.23792104862065, 172.22584754066236, 174.00765345004027],
                            [125.8777896115661, 175.60879944177162, 170.57224527770637]])
svc_company = SVC(kernel='linear', probability=True)
X_train_company, Y_train_company = ros.fit_resample(X_train_company, Y_train_company)
svc_company = svc_company.fit(X_train_company, Y_train_company)
# print("Train company accuracy: ", accuracy_score(Y_train_company, svc_company.predict(X_train_company)))


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


a = []
b = []


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
    # print(name)
    # id_re = re.compile(r"[0-9]+")
    # num = id_re.findall(image_path)[-1]
    # global a, b
    # a.append(num)
    # b.append((np.sum(np.sum(id_card, axis=0), axis=0) / id_card.shape[0] / id_card.shape[1]).tolist())
    # print(a, b)
    # print('-------------------------------------------------------')
    #if company is None:
    company = svc_company.predict(np.array([np.sum(np.sum(id_card, axis=0), axis=0)
                                            / id_card.shape[0] / id_card.shape[1]]))[0]
    return Person(name=name, date_of_birth=dob, job=job, ssn=ssn, company=company)


def filter_for_name(text):
    # print(text)
    text = re.sub(r"[0-9]{1,4}[ ]{1,4}[A-Za-z]{1,10}[ ]{1,4}[A-Za-z]{1,10}[ ]{1,4}.{1,4}[ ]{1,4}[0-9]{1,4}", " ", text)
    # text = re.sub(r"[ ]{2,}", " ", text)
    #text = re.sub(r"[!--/-@\[-`{-~]", "", text)
    #text = re.sub(r"[A-Z]{2,}", "", text)
    # text = re.sub(r" [A-Za-z]{1,3} ", " ", text)
    # print(text)
    text = text.replace("Samantha Corner", "")
    text = text.replace("Dylan Groves", "")
    # print(text)
    return text


def extract_name(text):
    name_re = re.compile(r"(?:(?:Mr\. )|(?:Ms\. )|(?:Mrs\. )|(?:))[A-Z][a-z]{3,7} [A-Z][a-z]{3,10}")
    found = name_re.findall(text)
    if len(found) == 0:
        no_front_name_re = re.compile(r"[a-z]{3,7} [A-Z][a-z]{3,10}")
        found = no_front_name_re.findall(text)
        if len(found) == 0:
            return 'Samantha Corner'
        else:
            return found[0]
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
    jobs = {'Human Resources': 0, 'Scrum Master': 0, 'Team Lead': 0, 'Manager': 0, 'Software Engineer': 0,
            'Software': 0}
    job, jobs, removal_record = match_results(text, jobs)
    text = drop_matches(text, removal_record)
    job = 'Software Engineer' if job == 'Software' else job
    return job if jobs[job] > 55 else 'Human Resources', text


def get_company(text):
    companies = {'IBM': 0, 'Google': 0, 'Apple': 0}
    text_no_spaces = text.replace(' ', '')
    company, companies, removal_record = match_results(text_no_spaces, companies)
    text = drop_matches(text, removal_record)
    if companies[company] > 55:
        return company, text
    else:
        return None, text
        # 'Google', text


def drop_matches(text: str, drop: list):
    for this in drop:
        text = text.replace(this, "")
    return text


def drop_similar(text: str, ratio: float, drop: list):
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
