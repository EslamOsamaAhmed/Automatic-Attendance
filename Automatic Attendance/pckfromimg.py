# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 04:05:28 2019

@author: eslam
"""

import cv2 
import cv2 as cv
import numpy as np
import scipy
import pickle
import os, os.path
from scipy import spatial 

##########################Face Detection################################
     
#face_cascade to Classifier the Face in the img
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read an Image to be Detected and Convert it to Array of Pixels
img = cv.imread("eslam.jpg")

#Convert Img to GrayScale Img Mode
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(gray)
cv2.imwrite('img_processing/face.jpg',equ)

Nimg = cv2.imread('img_processing/face.jpg')
median = cv2.medianBlur(Nimg, 5)
cv2.imwrite('img_processing/facefilternoise.jpg',median)

newimg = cv2.imread('img_processing/facefilternoise.jpg')

"""If faces are found, it returns the positions of detected faces as Rect(x,y,w,h)
Create ROI for the Face and Eye Detection
ROI Region of Interest"""

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
imgcounter = 0
for (x,y,w,h) in faces:
    cv.rectangle(newimg,(x,y),(x+w,y+h),(255,0,0),2)
    # cropped newimg 
    crop_img = newimg[y:y+h, x:x+w]
    crop_resize_img = cv2.resize(crop_img, (150, 150)) 
    cv2.imwrite("Cropped_IMG/" + str(imgcounter) + ".jpg", crop_resize_img)
    imgcounter = imgcounter + 1
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]

#cv.imshow('img',img)
cv2.imwrite("outputDetect.jpg", newimg)
############################ Feature Engineering #########################

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = cv.imread(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv.error as e:
        print ('Error: ' + str(e))
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path):
    result = {}
    result["face"] = extract_features(images_path)
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)
    fp.close()
    
crop_path = 'C:/Users/eslam/Desktop/GP/Cropped_IMG/'
cropped_num = len([f for f in os.listdir(crop_path)if os.path.isfile(os.path.join(crop_path, f))])

for facenum in range(cropped_num):
    batch_extractor("Cropped_IMG/0.jpg", "PCKs/0.pck")

