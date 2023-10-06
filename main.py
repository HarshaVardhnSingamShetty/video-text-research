
from PIL import Image
import pytesseract
# from wand.image import Image as Img
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np

import os
import cv2

image_frames = 'image_frames'


def files():
    try:
        os.remove(image_frames)
    except OSError:
        pass
    if not os. path.exists(image_frames):
        os.makedirs(image_frames)

    # specify the source video path
    src_vid = cv2. VideoCapture('d2.mp4')
    return (src_vid)


def process(src_vid):
    # Use an index to integer-name the files
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break
        # name each frame and save as png
        name = './image_frames/frame' + str(index) + '.png'
        # save every 100th frame
        if index % 100 == 0:
            print('Extracting frames...' + name)
            cv2.imwrite(name, frame)
        index += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    src_vid. release()
    cv2.destroyAllWindows()


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to make text stand out
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def get_text():
    for i in os. listdir(image_frames):
        print(str(i))
        my_example = Image.open(image_frames + "/" + i)

        # Preprocess the image
        preprocessed_image = preprocess_image(np.array(my_example))
        # Run OCR on the preprocessed image
        text = pytesseract.image_to_string(
            preprocessed_image, lang='eng', config='--psm 6 --oem 3')

        print("texxttt =", text)


vid = files()
print("🚀 ~ file: main.py:60 ~ vid:", vid)
process(vid)
get_text()
