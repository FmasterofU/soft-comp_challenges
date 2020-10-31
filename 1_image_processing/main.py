from process import count_blood_cells
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'train'+os.path.sep
# ------------------------------------------------------------------

processed_image_names = []
blood_cell_counts = []

for image_path in glob.glob(DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    blood_cell_counts.append(count_blood_cells(image_path))

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
# Kreiranje fajla sa rezultatima brojanja za svaku sliku
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s,%s,%s\n" % (image_name,
                                               blood_cell_counts[image_index][0],
                                               blood_cell_counts[image_index][1],
                                               blood_cell_counts[image_index][2])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)
# ------------------------------------------------------------------
