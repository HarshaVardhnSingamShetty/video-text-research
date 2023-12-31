
from PIL import Image
import pytesseract
# from wand.image import Image as Img
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import os
import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
image_frames = 'image_frames'


def files():
    try:
        os.remove(image_frames)
    except OSError:
        pass
    if not os. path.exists(image_frames):
        os.makedirs(image_frames)

    # specify the source video path
    src_vid = cv2. VideoCapture('d3.mp4')
    return (src_vid)


def process(src_vid):
    # Check if the 'image_frames' directory exists, and if it does, delete all files in it
    if os.path.exists(image_frames):
        for filename in os.listdir(image_frames):
            file_path = os.path.join(image_frames, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Use an index to integer-name the files
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break
        # name each frame and save as png
        name = './image_frames/frame' + str(index) + '.png'
        # save every 100th frame
        if index % 50 == 0:
            print('Extracting frames...' + name)
            cv2.imwrite(name, frame)
        index += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    src_vid. release()
    cv2.destroyAllWindows()


def preprocess_image(image, name):
    filename = './processed_images/' + str(name)
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    # processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)
    # _, processed_image = cv2.threshold(
    #     processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kernel = np.ones((3, 3), np.uint8)
    # processed_image = cv2.dilate(processed_image, kernel, iterations=1)
    # processed_image = cv2.erode(processed_image, kernel, iterations=1)

    cv2.imwrite(filename, processed_image)
    return processed_image


def clean_text(ocr_text):
    # Define a regular expression pattern to match the allowed characters
    pattern = r"[^A-Za-z0-9,.?!'$ ]"

    # Replace any characters that do not match the pattern with a space
    cleaned_text = re.sub(pattern, ' ', ocr_text)

    # Remove extra spaces and strip leading/trailing spaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text


def get_text_tesseract_ocr():
    for i in os. listdir(image_frames):
        print(str(i))
        my_example = Image.open(image_frames + "/" + i)

        # Preprocess the image
        preprocessed_image = preprocess_image(np.array(my_example), str(i))
        # Run OCR on the preprocessed image
        text = pytesseract.image_to_string(
            preprocessed_image, lang='eng', config='--psm 6 --oem 3')

        text = clean_text(text)

        print("texxttt =", text)


def get_text_easy_ocr():
    for i in os. listdir(image_frames):
        print("==============Text extracted from: ", str(i), "==============")

        IMAGE_PATH = os.path.abspath(os.path.join(image_frames, i))

        # IMAGE_PATH = 'frame200.png'
        reader = easyocr.Reader(['en'])
        result = reader.readtext(IMAGE_PATH)
        for k in range(len(result)):
            print(result[k][-2])


vid = files()
process(vid)

get_text_easy_ocr()


# get_text_tesseract_ocr()
