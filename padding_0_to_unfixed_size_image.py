'''
1. This file will recursivly read the image from Train_path and Test_path
2. Find the largest width of all images
3. Again recursivly read the image file and padding 0 to the image and resave it to original folder
'''

import os
import numpy as np
from PIL import Image
import cv2

#@width will be used as global variable
width =0

Train_path = os.getcwd()+'\\dataset\Image_unfixed\Train'
Test_path = os.getcwd()+'\\dataset\Image_unfixed\Test' 

#@getLargestWidthofImage is for get the largest with of whole images

def getLargestWidthofImage(f):
    global width
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            image  = cv2.imread(tmp_path)
            #print(tmp_path)
            if(width < image.shape[1]):
                width = image.shape[1]
        else:
            getLargestWidthofImage(tmp_path) 

#@re_largest_width for return the largest width of all images
def re_largest_width():
    global width
    getLargestWidthofImage(Train_path)
    getLargestWidthofImage(Test_path)
    return width

print(re_largest_width())

# padd 0 to the images in terms of resize the images to the max width
def padd_0(path,width):   
    image  = cv2.imread(path)
    temp_image = np.zeros((image.shape[0],width,image.shape[2]),dtype =image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                temp_image[i][j][k] =  image[i][j][k]
                
    cv2.imwrite(path, temp_image)
    print("pass")

#@padding_0_to_Image os for recursively read the image file and resave the processed image
def padding_0_to_Image(f,width):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            print(tmp_path)
            padd_0(tmp_path,width)
        else:
            padding_0_to_Image(tmp_path,width) 
    

padding_0_to_Image(Train_path,width)
padding_0_to_Image(Test_path,width)