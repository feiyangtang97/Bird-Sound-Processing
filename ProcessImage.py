'''
In this file
extractFeatures function to read the csv file and waves
pre_process is for randomly dvide the wav data and it's labels into train set and test set
spectrogram_generator is for generating spectrogram accoring to the data
generateImage is for saving the spectorgram into specific folder:

./dataset/Image_fixed/Train
||==same directory==|/Test		 
./dataset/Image_unfixed/Train
|| ==same directory== |/Test
./dataset/Image_with_axis/Train
|| ==same directory==  ||/Test 
Under the Train and Test folder, the folder name represent labels. 
For example: folder 0 present label is 0 

'''

import sys
import extractFeatures
import wave
import pylab
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from numpy import zeros, newaxis
import pandas as pd
import random
import numpy as np
import os,sys


#randomly pick around 20% of data in test set and 80% data in train set
def pre_process():
    data = extractFeatures.get_data()
    waves = extractFeatures.get_wav()
    #print(data)
    train_data = pd.DataFrame(columns=['id', 'mbe', 'label'])
    test_data = pd.DataFrame(columns=['id', 'mbe', 'label'])
    train_waves = pd.DataFrame(
    columns=['id', 'audio_data', 'sample_rate', 'sample_width', 'number_of_channels', 'number_of_frames'])
    test_waves = pd.DataFrame(
    columns=['id', 'audio_data', 'sample_rate', 'sample_width', 'number_of_channels', 'number_of_frames'])
    for i in range(data.shape[0]):
        if(random.randint(0,9)>=8):
            test_data = test_data.append(data.loc[i,],ignore_index=True)
            test_waves = test_waves.append(waves.loc[i,],ignore_index=True)           
        else:
            train_data = train_data.append(data.loc[i,], ignore_index=True)
            train_waves = train_waves.append(waves.loc[i,],ignore_index=True)
        
    return train_data,train_waves,test_data,test_waves



 #generate spectrogram and return the image maxtrix
def spectrogram_generator(input_data,index,newpath,Axes_and_colorbar,fixed):
    sound_info = input_data.iloc[index,1]
    frames = input_data.iloc[index,5]
    frame_rate = input_data.iloc[index,2]

    height = 4.8
    width =6.4
    if(fixed==False):
    	width = 0.071*frames*0.01;
    fig = plt.figure(num=None,figsize=(width,height)) 
    #plt.subplot(111)
    fig = plt.specgram(sound_info, NFFT=256, Fs=2, Fc=10,noverlap=128,
         cmap=None, xextent=None, pad_to=None, sides='default',
         scale_by_freq=None, mode='default', scale='default')
    #print(fig)
    if(Axes_and_colorbar==False):
        plt.axis('off')
    else:	
    	plt.colorbar()
    DIR = newpath 
    name = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    path = newpath+'\\'+str(name)+'.jpg'
    print(path)
    plt.tight_layout()
    if(Axes_and_colorbar==False):
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(path,bbox_inches='tight',pad_inches=0.0)
    plt.close('all')


train_data,train_waves,test_data,test_waves = pre_process()
print(test_data.shape[0]/train_data.shape[0])

#funtion del_file is for clear all the files and folders under the path
def  del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i) 
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
path1 = ''
path2 = ''

#function generateImage is for generator spectorgram accroding to waves and save them into folder
#@parameter Axes_and_colorbar states did the image have axes and colorbar
#@parameter parameter states generating fixed size of image or not
def generateImage(Axes_and_colorbar, fixed):
    if(Axes_and_colorbar==True):
        path1 = os.getcwd()+'\\'+'dataset\Image_with_axis\Train\\'
        path2 = os.getcwd()+'\\'+'dataset\Image_with_axis\Test\\'
    else:
        if(fixed==True):
            path1 = os.getcwd()+'\\'+'dataset\Image_fixed\Train\\'
            path2 = os.getcwd()+'\\'+'dataset\Image_fixed\Test\\'
        else:
            path1 = os.getcwd()+'\\'+'dataset\Image_unfixed\Train\\'
            path2 = os.getcwd()+'\\'+'dataset\Image_unfixed\Test\\'

    del_file(path1)
    del_file(path2)
#generate and save spectorgram train folder 
    for i in range(train_waves.shape[0]):         
        temp_label = train_data.iloc[i,2]
        print(temp_label)
        newpath = path1 +str(temp_label)  
        print (newpath)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        temp_data = spectrogram_generator(train_waves,i,newpath,Axes_and_colorbar,fixed)  
#generate and save spectorgram test folder 
    for i in range(test_waves.shape[0]):         
        temp_label = test_data.iloc[i,2]
        print(temp_label)
        newpath = path2+str(temp_label)  
        print (newpath)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        temp_data = spectrogram_generator(test_waves,i,newpath,Axes_and_colorbar,fixed)  
