'''
Methods to compare matched images
'''

from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv

def imshow(img):
    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

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


class PatchDiffCMP(ImageComparer):
    def __init__(self, patch_size=11, max_x=5, max_y=5, gaussian=False):
        '''
        Parameters
        ----------
        patch_size : int
            size of compared patches
        max_x : int
            maximum patch shift in the x coordinate
        max_y : int
            maximum patch shift in the y coordinate
        gaussian : boolean
            True to use gaussian kernel, False to use uniform kernel
        '''
        self.patch_size = patch_size
        self.max_x = max_x
        self.max_y = max_y
        self.gaussian = gaussian
        
        self.kernel = np.ones((patch_size, patch_size), dtype=np.float32)
        self.kernel /= patch_size * patch_size 

    def compare(self,img1,img2):
        '''
        Compare both images by subtracting them in a smart way,
        by finding most resembling patches in the neighborhood
        '''
        w, h = img1.shape[1], img1.shape[0]

        img1 = cv.GaussianBlur(img1,(11,11),0)
        img2 = cv.GaussianBlur(img2,(11,11),0)
        # Define translation
        M = np.array([
            [1,0,0],
            [0,1,0]
        ], dtype=np.double)

        result = 255.*np.ones((h,w))

        for dx in range(-self.max_x, self.max_x+1):
            for dy in range(-self.max_y, self.max_y+1):
                # Modify translation
                M[0,-1], M[1,-1] = dx, dy
                # Apply translation
                shifted = cv.warpAffine(img1, M, (w,h))
                # Compute distance
                diff = cv.absdiff(shifted,img2)
                diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
                diff = diff.astype(np.double)
                # Average over window
                diff = cv.filter2D(diff, -1, self.kernel) if not(self.gaussian) \
                       else cv.GaussianBlur(diff,(self.patch_size,self.patch_size),0)
                # Update result
                result = cv.min(result,diff)
        return np.uint8(result)

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