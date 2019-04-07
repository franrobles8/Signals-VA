# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:10:12 2019

@author: adgao
"""
import cv2 as cv
while True:
    m= cv.imread('./recortes_prueba/00047.ppm')
   
    cv.imshow('m', m)
    """
    mt= cv.imread('train/00399.ppm')
    cv.imshow('mt', mt)
    """
    if cv.waitKey(5) == 27:
            break
        
cv.destroyAllWindows()