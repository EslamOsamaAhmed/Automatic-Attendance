import cv2 
import cv2 as cv
import numpy as np
import scipy
import pickle
import time
import os, os.path
from scipy import spatial 


###########################Face Detection################################
     
##face_cascade to Classifier the Face in the img
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread("Your Image Path to be Recognized.jpg")

#Convert Img to GrayScale Img Mode
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(gray)
cv2.imwrite('c_img_processing/face.jpg',equ)

Nimg = cv2.imread('c_img_processing/face.jpg')
median = cv2.medianBlur(Nimg, 5)
cv2.imwrite('c_img_processing/filternoise.jpg',median)

newimg = cv2.imread('c_img_processing/filternoise.jpg')

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
    cv2.imwrite("Comparison_Imgs/" + str(imgcounter) + ".jpg", crop_resize_img)
    imgcounter = imgcounter + 1

cv2.imwrite("attendancedetect.jpg", newimg)

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

def cos_cdist(vector, comparisonImg):
    with open(comparisonImg, 'rb') as fp:
        data = pickle.load(fp)
        names = []
        matrix = []
        
        for k, v in data.items():
            names.append(k)
            matrix.append(v)
            
        matrix = np.asarray(matrix)
        names = np.asarray(names)
    fp.close()
    
    # getting cosine distance between search image and images database
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def match(image_path, comparisonImg, topn=5):
    features = extract_features(image_path)
    img_distances = cos_cdist(features, comparisonImg)
    
    return img_distances[0]

pck_path = 'C:/Users/eslam/Desktop/GP/PCKs/'
pck_num = len([f for f in os.listdir(pck_path)if os.path.isfile(os.path.join(pck_path, f))])

com_path = 'C:/Users/eslam/Desktop/GP/Comparison_Imgs/'
com_num = len([f for f in os.listdir(com_path)if os.path.isfile(os.path.join(com_path, f))])

attend = []

pcklist = np.array([3,4])
for cimg in range(com_num):
    for pck in range(len(pcklist)):
        print(1-match("Comparison_Imgs/" + str(cimg) + ".jpg", "PCKs/" + str(pcklist[pck]) + ".pck", topn=3))
           
        if (1-match("Comparison_Imgs/" + str(cimg) + ".jpg", "PCKs/" + str(pcklist[pck]) + ".pck", topn=3)  > 0.4):
                attend.append(pcklist[pck])
                
print(attend)
