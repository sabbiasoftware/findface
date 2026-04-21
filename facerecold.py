#!/usr/bin/python3
import face_recognition
import time
import numpy
import glob
import pathlib
import os
import pickle
from tqdm import tqdm
import tkinter
from PIL import Image as PILImage
from PIL import ImageTk as PILImageTk


def resize_maybe(image, maxdim):
    ratio = float(maxdim) / float(max(image.width, image.height))
    return image.resize((int(float(image.width) * ratio), int(float(image.height) * ratio)))


def rotate_maybe(image):
    exif = dict(image._getexif().items())
    orientation = 0
    if exif != None:
        if 0x0112 in exif.keys():
            orientation = exif[0x0112]
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    return image


def show_image(imgpath, rects=None, message=None):
    window = tkinter.Tk()
    window.title(message)

    canvas = tkinter.Canvas(window, width=1024, height=1024, bg="white")
    canvas.pack()

    pilimg = resize_maybe(rotate_maybe(PILImage.open(imgpath)), 1024)
    pilimgtk = PILImageTk.PhotoImage(pilimg)
    canvas.create_image((0, 0), image=pilimgtk, anchor="nw")

    if rects is not None:
        for y0, x1, y1, x0, c, t in rects:
            canvas.create_rectangle(x0, y0, x1, y1, outline=c, width=2)
            canvas.create_text(x0 + 4, y0 - 4, text=t, anchor="sw", fill=c)

    window.bind("<Escape>", lambda e: window.destroy())

    window.mainloop()


def show_images(imgpath1, imgpath2, rect1=None, rect2=None, message=""):
    window = tkinter.Tk()
    window.title(message)

    window.columnconfigure(0, weight=1)
    window.columnconfigure(1, weight=1)
    window.rowconfigure(0, weight=1)
    window.rowconfigure(1, weight=0)

    canvas1 = tkinter.Canvas(window, width=1024, height=1024, bg="white")
    canvas1.grid(column=0, row=0)

    canvas2 = tkinter.Canvas(window, width=1024, height=1024, bg="white")
    canvas2.grid(column=1, row=0)

    label1 = tkinter.Label(window, text=imgpath1)
    label1.grid(column=0, row=1)

    label2 = tkinter.Label(window, text=imgpath2)
    label2.grid(column=1, row=1)

    pilimg1 = resize_maybe(rotate_maybe(PILImage.open(imgpath1)), 1024)
    pilimgtk1 = PILImageTk.PhotoImage(pilimg1)
    canvas1.create_image((0, 0), image=pilimgtk1, anchor="nw")

    pilimg2 = resize_maybe(rotate_maybe(PILImage.open(imgpath2)), 1024)
    pilimgtk2 = PILImageTk.PhotoImage(pilimg2)
    canvas2.create_image((0, 0), image=pilimgtk2, anchor="nw")

    if rect1 != None:
        y0, x1, y1, x0 = rect1
        canvas1.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    if rect2 != None:
        y0, x1, y1, x0 = rect2
        canvas2.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    window.bind("<Escape>", lambda e: window.destroy())

    window.mainloop()


def load_image(path):
    # image = PILImage.open(path)

    # exif = dict(image._getexif().items())

    # maxsize = max(image.size)
    # ratio = maxsize / 1024.0
    # image = image.resize((int(image.size[0] / ratio), int(image.size[1] / ratio)), PILImage.ANTIALIAS)

    # orientation = 0
    # if exif != None:
    #     if 0x0112 in exif.keys():
    #         orientation = exif[0x0112]
    # #print(exif[0x0112])
    # if orientation == 3:
    #     image=image.rotate(180, expand=True)
    # elif orientation == 6:
    #     image=image.rotate(270, expand=True)
    # elif orientation == 8:
    #     image=image.rotate(90, expand=True)

    image = resize_maybe(rotate_maybe(PILImage.open(path)), 1024)
    image = numpy.array(image)
    return image


# def load_known_faces(path):
#     kfe = []
#     imagepathlist = []
#     imagepathlist.extend(glob.glob(path + "/*.jpg"))
#     imagepathlist.extend(glob.glob(path + "/*.JPG"))
#     for imagepath in imagepathlist:
#         encfound = False
#         for encpath in glob.glob(str(pathlib.PurePosixPath(imagepath).with_suffix("")) + ".*.npy"):
#             encfound = True
#             kfe.append((pathlib.PurePosixPath(encpath).stem, numpy.load(encpath)))

#         if not encfound:
#             image = load_image(imagepath)
#             #face_locations = face_recognition.face_locations(image)
#             face_encodings = face_recognition.face_encodings(image)
#             index = 1
#             for face_encoding in face_encodings:
#                 kfe.append((pathlib.PurePosixPath(imagepath).stem, face_encoding))
#                 numpy.save(str(pathlib.PurePosixPath(imagepath).with_suffix(".{}.npy".format(index))), face_encoding)
#                 index = index + 1

#     with open("known_face_encodings.pickle", "wb") as kfe_file:
#         pickle.dump(kfe, kfe_file)

#     return kfe

# def scan_directory(directory):
#     for imagepath in glob.glob(directory + "/**/*.JPG", recursive=True):
#         image = load_image(imagepath)
#         face_encodings = face_recognition.face_encodings(image)
#         for face_encoding in face_encodings:
#             mindist = 999999.9
#             minknown = ""
#             for known_face_encoding in known_face_encodings:
#                 dist = face_recognition.face_distance([known_face_encoding[1]], face_encoding)[0]
#                 #print("{} - {}".format(dist, known_face_encoding[0]))
#                 if dist < mindist:
#                     mindist = dist
#                     minknown = known_face_encoding[0]
#             print("{},{},{}".format(imagepath,mindist, minknown))


def open_index(dir, skiprefresh=False):
    index = dict()
    indexpath = os.path.join(dir, "facerec.index")

    if os.path.exists(indexpath):
        print("Opening index {}".format(indexpath))
        with open(indexpath, "rb") as indexfile:
            index = pickle.load(indexfile)
    else:
        print("No index found at {}".format(indexpath))

    print("Found {} indexes".format(len(index)))
    was_update = False
    if not skiprefresh:
        for imagepath in tqdm(glob.glob(dir + "/**/*.JPG", recursive=True)):
            try:
                if imagepath not in index.keys():
                    # tqdm.set_description("Indexing {}".format(imagepath))
                    # print("Indexing {}".format(imagepath))
                    img = load_image(imagepath)
                    face_locs = face_recognition.face_locations(img)
                    face_encs = face_recognition.face_encodings(img, known_face_locations=face_locs)

                    if len(face_locs) != len(face_encs):
                        print("Lengths do not equal {}".format(imagepath))

                    index[imagepath] = [
                        (face_locs[i], face_encs[i]) for i in range(0, max(len(face_locs), len(face_encs)))
                    ]
                    was_update = True
            except Exception:
                continue

        if was_update:
            with open(indexpath, "wb") as indexfile:
                pickle.dump(index, indexfile)

    return index


#
# { img_path : [ ( (y0, x1, y1, x0), [e0, e1, ..., e127] ) ] }
#                  +--------------+  +-----------------+
#                      face_loc           face_enc
#


def search(known_index, image_index, minFaces, maxFaces, incFaces, excFaces, top):
    found = 0
    for path, index in image_index.items():
        # skip image if it does not have at least minFaces faces
        if len(index) < minFaces:
            continue

        # skip imate if it has more than maxFaces faces
        if len(index) > maxFaces:
            continue

        # list to gather rects to draw in case image is a match
        rects = []

        face_found = False

        # check each face in image
        for image_face_loc, image_face_enc in index:
            face_found = False

            # find best matching known face
            min_dist = None
            min_known_face_path = None

            for known_face_path in known_index.keys():
                known_face_loc, known_face_enc = known_index[known_face_path][0]
                dist = face_recognition.face_distance([known_face_enc], image_face_enc)[0]
                if dist < 0.6 and dist < min_dist:
                    min_dist = dist
                    min_known_face_path = known_face_path

            # check if best matching face is one of the required faces
            if not (min_dist is None or min_known_face_path is None):
                for incFace in incFaces:
                    if incFace.lower() in pathlib.Path(min_known_face_path).stem.lower():
                        face_found = True
                        break

                if face_found:
                    # best matching face is one of the required faces
                    rects.append(
                        (
                            image_face_loc[0],
                            image_face_loc[1],
                            image_face_loc[2],
                            image_face_loc[3],
                            "red",
                            "{}: {:.2f}".format(pathlib.Path(min_known_face_path).stem, min_dist),
                        )
                    )

            else:
                break

        if face_found:
            # yaay, all faces are one of the required faces
            print(path)
            # show_image(path, message="It's a match!", rects=rects)
            found += 1
            if found >= top:
                print("Found {} images, exiting".format(top))
                break


# img = load_image("/home/ssuranyi/Pictures/2022-08-12/IMG_1961.JPG")
# face_locs = face_recognition.face_locations(img)

# for face_loc in face_locs:
#     show_images("/home/ssuranyi/Pictures/2022-08-12/IMG_1961.JPG", "/home/ssuranyi/Pictures/2022-08-12/IMG_1942.JPG", rect1=face_loc)

known_index = open_index("/home/ssuranyi/Pictures/KNOWN_PEOPLE_FOLDER")
image_index = open_index("/run/media/ssuranyi/My Passport 4T/Kép", True)

# search(known_index, image_index, 4, 4, ["Julcsi", "Szabolcs", "Bence", "Áron"], [], 10)
search(known_index, image_index, 1, 99, ["Peti"], [], 1000)

# known_face_encodings = load_known_faces("/home/ssuranyi/Pictures/KNOWN_PEOPLE_FOLDER")
# scan_directory("/home/ssuranyi/Pictures/_")

# image = load_image("/home/ssuranyi/Pictures/_/2011-12-28/IMG_5508.JPG")
# face_encodings = face_recognition.face_encodings(image)
# for face_encoding in face_encodings:
#     mindist = 999999.9
#     minknown = ""
#     print()
#     for known_face_encoding in known_face_encodings:
#         dist = face_recognition.face_distance([known_face_encoding[1]], face_encoding)[0]
#         #print("{} - {}".format(dist, known_face_encoding[0]))
#         if dist < mindist:
#             mindist = dist
#             minknown = known_face_encoding[0]
#     print("{} - {}".format(mindist, minknown))
