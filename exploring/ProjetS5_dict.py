import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
from pytesseract import Output


def clean_dict(d) :
#input: pytesseract dict
#output: dict without the values for which the confiance value is smaller than 60%
    i=0
    j=0
    while i < len(d['conf']):
        if int(d['conf'][i]) <= 60 :
            d['level'].pop(i)
            d['page_num'].pop(i)
            d['block_num'].pop(i)
            d['par_num'].pop(i)
            d['line_num'].pop(i)
            d['word_num'].pop(i)
            d['left'].pop(i)
            d['top'].pop(i)
            d['width'].pop(i)
            d['height'].pop(i)
            d['conf'].pop(i)
            d['text'].pop(i)
            j += 1
            i -= 1
        i += 1
    return d


def create_dict()
    text = cv2.imread('../data/test1.jpg')
    text_rgb = cv2.cvtColor(text, cv2.COLOR_BGR2RGB)
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    d = pytesseract.image_to_data(text_rgb,output_type=Output.DICT)
    return clean_dict(d)


