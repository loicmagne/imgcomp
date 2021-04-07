'''
Methods to compare matched images
'''

from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv

class ImageComparer(ABC):
    @abstractmethod
    def compare(self,src,tar):
        '''
        Methods which takes two image matched using correspondences
        and output a probability map over the second image which tells
        how locally different the second image is.
        
        Parameters
        ----------
        img1,img2 : np.ndarray

        Returns
        -------
        diff : np.ndarray
        '''
        pass

class RawDiffCMP(ImageComparer):
    def __init__(self):
        self.kernel = kernel = np.ones((5,5),np.uint8)
        
    def compare(self,img1,img2):
        '''
        Compare both images by directly subtracting them
        '''
        img1_blur = cv.GaussianBlur(img1,(21,21),0)
        img2_blur = cv.GaussianBlur(img2,(21,21),0)
        diff = cv.subtract(img1_blur,img2_blur)
        
        diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        diff = cv.dilate(diff, self.kernel, iterations = 2)
        
        _,diff = cv.threshold(diff,50,255,cv.THRESH_TOZERO)
        
        return np.uint8(diff)

class BGSubCMP(ImageComparer):
    def __init__(self):
        self.backSub = cv.createBackgroundSubtractorKNN(history = 2)
        self.kernel = np.ones((5,5),np.uint8)

    def compare(self,img1,img2):
        '''
        Compare both images by using background subtraction
        techniques
        '''
        mask = self.backSub.apply(img1)
        mask = self.backSub.apply(img1)
        mask = self.backSub.apply(img1)
        mask = self.backSub.apply(img1)
        mask = self.backSub.apply(img1)
        mask = self.backSub.apply(img2)
        
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel, iterations=1)

        return np.uint8(mask)

class OptFlowCMP(ImageComparer):
    def compare(self,img1,img2):
        '''
        Compare both images by using optical flow techniques
        '''
        gimg1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        gimg2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
        
        result = np.zeros_like(img1)
        result[...,1] = 255
        
        flow = cv.calcOpticalFlowFarneback(gimg1,gimg2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        result[...,0] = ang*180/np.pi/2
        result[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        result = cv.cvtColor(result,cv.COLOR_HSV2BGR)
        return result