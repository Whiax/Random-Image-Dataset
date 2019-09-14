# -*- coding: utf-8 -*-
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

#methods
#random integer
def irunif(mi,ma):
    return np.random.randint(min(mi,ma), max(mi,ma))
ir = irunif
rc = random.choice

#random batch
def getBatch(batch_size=50, image_size=300, minsize=8, maxsize=80, verbose=0, display=0):
    #init
    pad = 20
    channel_size = 3
    images = np.zeros((batch_size,image_size,image_size,channel_size))
    boxes = list()
    
    #batch creation, for each image in the batch
    tic = int(round(time.time() * 1000))
    for img in images:
        n_point = ir(3,5)
        n_object = ir(3,50)
        boxes.append(np.zeros((n_object,4,1,2),dtype=np.int32))
        #for each object to draw
        for o in range(n_object):
            #build image
            cx,cy = ir(pad,image_size-1-pad),ir(pad,image_size-1-pad)
            sx,sy = ir(minsize,maxsize),ir(minsize,maxsize)
            rpoints = [(cx+ir(minsize-1,sx)*rc((-1,1)),cy+ir(minsize-1,sy)*rc((-1,1))) for i in range(n_point)]
            ctr = np.array(rpoints).reshape((-1,1,2)).astype(np.int32)
            r,g,b = irunif(0,255),irunif(0,255),irunif(0,255)
            c = (r,g,b)
            cv2.drawContours(img, [ctr], 0, c, -1)
            
            #build box
            ctr = ctr[:,0,:]
            topleft = (min(ctr[:,0]),min(ctr[:,1]),)
            topright = (min(ctr[:,0]),max(ctr[:,1]),)
            bottomright = (max(ctr[:,0]),max(ctr[:,1]),)
            bottomleft = (max(ctr[:,0]),min(ctr[:,1]),)
            boxes[-1][o] = np.array([topleft,topright,bottomright,bottomleft]).reshape(-1,1,2)
            #cv2.drawContours(img, [boxes[-1][o]], 0, c, -1)
    toc = int(round(time.time() * 1000))
    delay = toc-tic
    
    #display time to generate batch
    if verbose:
        print("Generated in ", delay,'ms')
        
    #display first image
    if display:
        plt.imshow(images[0])
        plt.show()
        
    #return everything
    return images,boxes










