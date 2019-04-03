# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:10:12 2019

@author: adgao
"""
import cv2 as cv
while True:
    m= cv.imread('./train_recortadas/35/00000.ppm')
    mt= cv.imread('train/00000.ppm')
    cv.imshow('m', m)
    cv.imshow('mt', mt)
    if cv.waitKey(5) == 27:
            break
        
cv.destroyAllWindows()