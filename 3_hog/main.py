from process import train_or_load_age_model, train_or_load_gender_model, train_or_load_race_model,\
    predict_age, predict_gender, predict_race
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep
# -------------------------------------------------------------------

# indeksiranje labela za brzu pretragu
labeled_age = dict()
labeled_gender = dict()
labeled_race = dict()
with open(TRAIN_DATASET_PATH+'annotations.csv', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index > 0:
            cols = line.replace('\n', '').split(',')
            image = cols[0].replace('\r', '')
            age = cols[1].replace('\r', '')
            gender = cols[2].replace('\r', '')
            race = cols[3].replace('\r', '')
            labeled_age[str(image)] = int(age)
            labeled_gender[str(image)] = gender
            labeled_race[str(image)] = race

# priprema skupa podataka za treniranje modela
train_image_paths = []
train_age_labels = []
train_gender_labels = []
train_race_labels = []

for image_name in os.listdir(TRAIN_DATASET_PATH):
    if '.jpg' in image_name:
        train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, image_name))
        train_age_labels.append(labeled_age[image_name])
        train_gender_labels.append(labeled_gender[image_name])
        train_race_labels.append(labeled_race[image_name])

# istrenirati modele za prepoznavanje godine, pol i rasu
model_age = train_or_load_age_model(train_image_paths, train_age_labels)
model_gender = train_or_load_gender_model(train_image_paths, train_gender_labels)
model_race = train_or_load_race_model(train_image_paths, train_race_labels)

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka, koriscenjem istreniranog modela
processed_image_names = []
predicted_age = []
predicted_gender = []
predicted_race = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    # run predictions
    predicted_age.append(predict_age(model_age, image_path))
    predicted_gender.append(predict_gender(model_gender, image_path))
    predicted_race.append(predict_race(model_race, image_path))


# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s,%s,%s\n" % (image_name, predicted_age[image_index], predicted_gender[image_index], predicted_race[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
