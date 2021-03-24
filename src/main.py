import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew

import pytesseract
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RANSACRegressor
import seaborn as sns

from os import listdir
from os.path import isfile, join
import imutils
import warnings 
warnings.filterwarnings("ignore")

from utils import Preprocess, Blind_image_adjustment
import argparse

if __name__ == '__main__':

    # Retrieve command line arguments
    parser = argparse.ArgumentParser(description="Program for blind image adjustment. Program use is as follows: python3 main.py image_path. E.g. python3 main.py path='../data/screens_phone/screen_5.jpeg'")
    parser.add_argument("--path", type=str, default='../data/screens_phone/screen_5.jpeg', help="Path of the image")
    parser.add_argument("--save_path", type=str, default='', help="Path to save the image")

    args = parser.parse_args()
    path = args.path
    save = args.save_path
    img = cv2.imread(path) #reading the image

    ##### Preprocessing
    preprocessor = Preprocess()
    img =  preprocessor.remove_shadows(img.copy())
    angle_desk,img  = preprocessor.deskew(img.copy())
    edges = preprocessor.edge_detection(img.copy())
    lines = preprocessor.line_detection(edges)
    lines = preprocessor.filter_hough_lines(lines,edges)

    ##### Computing lines and angles
    angles_deg = preprocessor.compute_angles(lines)
    rot_angles = preprocessor.adjust_angles(angles_deg)
    lines_candidates = preprocessor.filter_lines_direction(lines,rot_angles,angle_desk)
   
    ##### adjusting blind image
    blind_image_adjustor = Blind_image_adjustment(img,lines_candidates,img.shape[0])
    _ = blind_image_adjustor.find_perfect_rectangle()
    edge_lines = blind_image_adjustor.compute_edge_lines()
    _= blind_image_adjustor.find_corners(edge_lines)
    im_out = blind_image_adjustor.adjust_image(margin=100)

    plt.imshow(im_out)
    
    cv2.imwrite(join(save,'result.png'), im_out)
