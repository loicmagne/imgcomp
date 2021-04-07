'''
Methods to compute correspondences between pairs of images
'''
import numpy as np
import cv2 as cv

class ImageMatcher():
    def __init__(self):
        '''
        Class to compute features of a pair of images, match them, and
        compute a homography between the two images
        '''
        self.MIN_MATCHES = 4
        self.sift = cv.SIFT_create()
        self.matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        
    def detectKeypoints(self,image):
        '''
        Computes the keypoints of an image and their descriptors
        '''
        gimg = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gimg,None)
        return kp, des

    def matchKeypoints(self,kp1,kp2,des1,des2,threshold=5.0):
        '''
        Match keypoints from two images, returns the matches, the matched and unmatched
        keypoints for both images, and the homography matrix
        
        Parameters
        ----------
        kp1, kp2 : cv.KeyPoint list
        des1, des2 : numpy.ndarray

        Returns
        -------
        H : np.array
            Homography matrix
        '''
        matches = self.matcher.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        kp1_matched = [kp1[m.queryIdx] for m in matches]
        kp2_matched = [kp2[m.trainIdx] for m in matches]
        
        if len(matches) > self.MIN_MATCHES:
            pts1 = np.float32([kp.pt for kp in kp1_matched]).reshape(-1, 1, 2)
            pts2 = np.float32([kp.pt for kp in kp2_matched]).reshape(-1, 1, 2)
            H,_ = cv.findHomography(pts2, pts1, cv.RANSAC, threshold)
            return H
        
        else:
            raise Error('Images not similar enough')

    def match(self,src,tar):
        '''
        Match two images
        
        Parameters
        ----------
        src : cv.Mat
            reference image
        tar : cv.Mat
            target image to be compared with the reference

        Returns
        -------
        src : cv.Mat
            reference image
        result : cv.Mat
            target image warped into the reference image
            coordinates

        '''
        w1, h1 = src.shape[1], src.shape[0]
        w2, h2 = tar.shape[1], tar.shape[0]
        
        # Find keypoints and their descriptors
        kp1, des1 = self.detectKeypoints(src)
        kp2, des2 = self.detectKeypoints(tar)
        
        # Match keypoints
        H = self.matchKeypoints(kp1,kp2,des1,des2)
        
        # Transform the target image 
        result = cv.warpPerspective(tar, H, (w1, h1))
        
        return src, result