import argparse
import face_recognition
import numpy
import glob
import pathlib
import os
import pickle
from tqdm import tqdm
import tkinter
from PIL import Image as PILImage
from PIL import ImageTk as PILImageTk
import random
import shutil
import traceback


def resize_maybe(image, maxdim):
    ratio = float(maxdim) / float(max(image.width, image.height))
    return image.resize((int(float(image.width) * ratio), int(float(image.height) * ratio)))


def rotate_maybe(image):
    exif = image._getexif()
    if exif is None:
        return image
    exifdict = dict(exif.items())
    orientation = 0
    if exifdict is not None:
        if 0x0112 in exifdict.keys():
            orientation = exifdict[0x0112]
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    return image


def load_image(path):
    image = resize_maybe(rotate_maybe(PILImage.open(path)), 1024)
    image = numpy.array(image)
    return image


INDEX_FILENAME = "facerec.idx"


def open_index(dir):
    index = dict()
    index_path = os.path.join(dir, INDEX_FILENAME)
    if os.path.exists(index_path):
        print("Opening index {}".format(index_path))
        with open(index_path, "rb") as indexfile:
            index = pickle.load(indexfile)
        print("Found {} index entries".format(len(index)))
    else:
        print("No index found at {}".format(index_path))
    return index


def save_index(dir, index):
    index_path = os.path.join(dir, INDEX_FILENAME)
    with open(index_path, "wb") as indexfile:
        pickle.dump(index, indexfile)


# def relpath_index(dir, index):
#     index_new = dict()
#     for path, data in index.items():
#         relpath = "/run" + path
#         try:
#             relpath = pathlib.Path(relpath).relative_to(dir)
#         except ValueError:
#             pass
#         index_new[relpath] = data
#     return index_new


def calc_index_entry(dir, imgpath):
    index_entry = None
    try:
        img = load_image(pathlib.Path(dir).joinpath(imgpath))
        face_locs = face_recognition.face_locations(img)
        face_encs = face_recognition.face_encodings(img, known_face_locations=face_locs)

        if len(face_locs) != len(face_encs):
            print("Lengths do not equal {}".format(imgpath))

        index_entry = [(face_locs[i], face_encs[i]) for i in range(0, max(len(face_locs), len(face_encs)))]
    except Exception as e:
        traceback.print_exc()
        pass
    return index_entry


def compare_index_entries(ie1, ie2):
        if ie1 is None and ie2 is None:
            return True
        if ie1 is None or ie2 is None:
            return True
        if len(ie1) != len(ie2):
            return True

        match = True
        for i in range(len(ie1)):
            locs_match = numpy.array_equal(ie1[i][0], ie2[i][0])
            if ie1[i][1] is not None and ie2[i][1] is not None:
                encs_match = numpy.allclose(ie1[i][1], ie2[i][1])
            else:
                encs_match = (ie1[i][1] is None) == (ie2[i][1] is None)
            if not (locs_match and encs_match):
                match = False
                break
        return match

def test_index(dir, index):
    TEST_COUNT = 10
    print("Testing index for {} random entries".format(TEST_COUNT))
    for _ in range(0, TEST_COUNT):
        imgpath, index_entry = random.choice(list(index.items()))
        index_entry_new = calc_index_entry(dir, imgpath)
        match = compare_index_entries(index_entry, index_entry_new):
        print("Index {} for {}".format("ok" if match else "mismatch", imgpath))


def update_index_remove_non_existing(dir, index):
    imgpaths_to_remove = []
    for imgpath in index.keys():
        if not os.path.exists(os.path.join(dir, imgpath)):
            imgpaths_to_remove.append(imgpath)
    for imgpath in imgpaths_to_remove:
        index.pop(imgpath)
    return len(imgpaths_to_remove) > 0


def update_index_add_new(dir, index):
    try:
        print("Scanning directory {}".format(dir))
        imgpaths_to_index = []
        for p in glob.glob(dir + "**", recursive=True):
            path = pathlib.Path(p)
            if path.is_file and path.suffix.lower() in [".jpg", ".jpeg" ".png"]:
                relpath = path.relative_to(dir)  # should not raise ValueError, right?
                if relpath not in index.keys():
                    imgpaths_to_index.append(relpath)
        if len(imgpaths_to_index) == 0:
            print("Index up-to-date")
            return True

        print("Found {} images to index".format(len(imgpaths_to_index)))
        for imgpath in tqdm(imgpaths_to_index):
            index_entry = calc_index_entry(dir, imgpath)
            if index_entry is not None:
                index[imgpath] = index_entry
        return True
    except KeyboardInterrupt:
        print("Index build aborted")
        return False


def update_index(dir, index):
    removed = update_index_remove_non_existing(dir, index)
    added = update_index_add_new(dir, index)
    return removed or added


def known_face_name_match(known_imgpath, name):
    kip = known_imgpath.lower()
    prefix1 = name.lower() + "_"
    prefix2 = name.lower() + "."
    return kip.startswith(prefix1) or kip.startswith(prefix2)


# Search for face by method 1
# - look for known faces that match name in parameter
# - look for faces that are closer to any of the matching known faces by threshold
def search_face_1(index_known, index_images, name, threshold):
    found_images = []

    known_imgpaths = []
    for known_imgpath in index_known.keys():
        if known_face_name_match(known_imgpath, name):
            known_imgpaths.append(known_imgpath)

    if len(known_imgpaths) == 0:
        print("No known face found matching {}".format(name))
        return

    for imgpath, index_entry in index_images.items():
        for _, face_enc in index_entry:
            for known_imgpath in known_imgpaths:
                known_faces = index_known[known_imgpath]
                for _, known_face_enc in known_faces:
                    dist = face_recognition.face_distance([known_face_enc], face_enc)[0]
                    if dist <= threshold:
                        found_images.append(imgpath)
                        print("{}: {} ({})".format(dist, imgpath, known_imgpath))

    return found_images


# Search for face by method 2
# - look for closest face among known faces
# - consider face found if known face name matches name in parameter and distance is less or equal than threshold
def search_face_2(index_known, index_images, name, threshold):
    found_images = []
    for imgpath, index_entries in index_images.items():
        for _, face_enc in index_entries:
            closest_imgpath = None
            closest_dist = None

            for known_imgpath, known_index_entries in index_known.items():
                for _, known_face_enc in known_index_entries:
                    dist = face_recognition.face_distance([known_face_enc], face_enc)[0]
                    if closest_dist is None or closest_dist > dist:
                        closest_imgpath = known_imgpath
                        closest_dist = dist
            if closest_dist > threshold:
                # Even the best match is farther than threshold, proceed to next image
                continue

            if known_face_name_match(closest_imgpath, name):
                # Best match is at acceptable distance but does not match face looked for
                continue

            found_images.append(imgpath)
            print("{}: {} ({})".format(closest_dist, imgpath, closest_imgpath))
    return found_images


def search_face(index_known, index_images, name, threshold):
    return search_face_2(index_known, index_images, name, threshold)


# dir = "/home/ssuranyi/Pictures/facerectest/"
img_dir = "/run/media/ssuranyi/My Passport 4T/Kép/"
index = open_index(img_dir)
# update_index(img_dir, index)
save_index(img_dir, index)

known_dir = "/home/ssuranyi/Pictures/KNOWN_PEOPLE_FOLDER/"
index_known = open_index(known_dir)
update_index(known_dir, index_known)
save_index(known_dir, index_known)

for imgpath, data in index_known.items():
    print(imgpath)

dest_dir = "/home/ssuranyi/Pictures/facerecout/"
imgs = search_face(index_known, index, "Peti", 0.6)

if pathlib.Path(dest_dir).is_dir():
    shutil.rmtree(dest_dir)
os.mkdir(dest_dir)
for img in tqdm(imgs):
    imgpath = os.path.join(img_dir, img)
    shutil.copy(imgpath, dest_dir)

# test_index(img_dir, index)
# save_index(img_dir, index)








parser = argparse.ArgumentParser("findface - search for images in a directory hierarchy that contain faces specified")
parser.add_argument(
    "-q",
    "--quick",
    action="store_true",
    help="Quick find, skip updating index. When searching in a directory the first time, make sure to use facefind WITHOUT this paramater, to allow building the initial index.",
)
parser.add_argument(
    "-F",
    "--faces-dir",
    default="FF_FACES",
    help="Override default face directory of <dir>/FF_FACES. If relative, then relative to <dir>.",
)
parser.add_argument(
    "-c",
    "--copy",
    action="store_true",
    help="Copy images found to a designated destination folder. Default destination directory is <dir>/FF_FACES_FOUND, use -C to override.",
)
parser.add_argument(
    "-C",
    "--copy-dir",
    default="FF_FACES_FOUND",
    help="Override default destination folder of <dir>/FF_FACES_FOUND. If relative, then relative to <dir>.",
)
parser.add_argument(
    "-b",
    "--brief",
    action="store_true",
    help="Reduce verbosity and print only path of found images.",
)
parser.add_argument(
    "-t",
    "--tolerance",
    type=float,
    default=0.6,
    help="Override default tolerance of 0.6. Lower tolerance value results less faces found.",
)
parser.add_argument(
    "-m",
    "--method",
    type=int,
    choices=[0, 1],
    help="Choose search method (0 or 1). 0: search for images that have at least one face that has a distance less or equal to tolerance. 1 (default): search for images that have at least one face that has a distance less or equal to tolerance AND no other face in <faces-dir> has less distance.",
)

parser.add_argument(
    "directory",
    help="Directory to search within recursively.",
)
args = parser.parse_args()
