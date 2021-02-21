import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from PIL import Image

class Error(Exception):
    pass

class ImageComparator():
    def __init__(self):
        '''
        Class to compare two images
        '''
        self.MIN_MATCHES = 4
        
    def detectKeypoints(self,image):
        '''
        Computes the keypoints of an image and their descriptors
        
        Parameters
        ----------
        image : opencv RGB image
        '''
        gimg = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(gimg,None)
        return kp, des

    def matchKeypoints(self,kp1,kp2,des1,des2,threshold=5.0):
        '''
        Match keypoints from two images, returns the matches, the matched and unmatched
        keypoints for both images, and the homography matrix
        
        Parameters
        ----------
        kp1, kp2 : cv.KeyPoint list
        des1, des2 : numpy.ndarray
        '''
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        kp1_matched = [kp1[m.queryIdx] for m in matches]
        kp2_matched = [kp2[m.trainIdx] for m in matches]

        kp1_miss_matched = [kp for kp in kp1 if kp not in kp1_matched]
        kp2_miss_matched = sorted([kp for kp in kp2 if kp not in kp2_matched], key = lambda x:x.response)
        
        if len(matches) > self.MIN_MATCHES:
            pts1 = np.float32([kp.pt for kp in kp1_matched]).reshape(-1, 1, 2)
            pts2 = np.float32([kp.pt for kp in kp2_matched]).reshape(-1, 1, 2)
            H, status = cv.findHomography(pts2, pts1, cv.RANSAC, threshold)
            return matches,kp1_matched,kp2_matched,kp1_miss_matched,kp2_miss_matched,H,status
        
        else:
            raise Error('Images not similar enough')
        
    def probabilityMap(self,kp1,kp2,img_size,tile_size,diff_threshold=5):
        '''
        Output a probability grid where each cell gives the probability
        that there is a difference between the two images in the given
        cell
        
        Parameters
        ----------
        kp1, kp2 : cv.KeyPoint list
        img_size : (int,int)
        tile_size : int
        diff_threshold : int
        '''
        w,h = img_size
        x = math.ceil(w/tile_size)
        y = math.ceil(h/tile_size)
        t1 = np.zeros((y,x))
        t2 = np.zeros((y,x))
        for k in kp1:
            t1[int(k.pt[1]//tile_size),int(k.pt[0]/tile_size)] += 1
        for k in kp2:
            t2[int(k.pt[1]//tile_size),int(k.pt[0]/tile_size)] += 1

        t = (t2-t1)
        t[t<=5] = 0
        t /= (t1+t2+10e-3)
        t = cv.resize(t,img_size)[:h,:w]
        
        # bins = np.linspace(t.min(),t.max(), 50)
        # plt.xlim([t.min(), t.max()])
        # plt.hist(t.flatten(), bins=bins)
        # plt.show()
        return np.uint8(255 * t)
        
    def compare(self,src,trgt,dest):
        '''
        Compare two images
        
        Parameters
        ----------
        src : string
            path to the reference image
        trgt : string
            path to the target image to be compared with the reference
        dest : string
            path where to save the comparison
        '''
        src = cv.imread(src)
        trgt = cv.imread(trgt)
        w1, h1 = src.shape[1], src.shape[0]
        w2, h2 = trgt.shape[1], trgt.shape[0]
        
        kp1, des1 = self.detectKeypoints(src)
        kp2, des2 = self.detectKeypoints(trgt)
        
        
        matches,\
        kp1_matched,kp2_matched,\
        kp1_miss_matched,kp2_miss_matched,\
        H,status\
        = self.matchKeypoints(kp1,kp2,des1,des2)
        
        # transform the target image 
        result = cv.warpPerspective(trgt, H, (w1, h1))
        
        # get the new position of the keypoints of the target image
        kp2_pt = np.array([kp.pt for kp in kp2])
        # homogeneous coordinate
        kp2_pt = np.concatenate((kp2_pt,np.ones((kp2_pt.shape[0],1))),axis=1)
        # apply the homography
        kp2_new_pt = np.dot(H,kp2_pt.T).T
        # divide by the last component
        kp2_new_pt = kp2_new_pt[:,:2]/kp2_new_pt[:,2][...,np.newaxis]
        kp2_new_pt = kp2_new_pt[(kp2_new_pt[:,0]>=0)\
                               &(kp2_new_pt[:,1]>=0)\
                               &(kp2_new_pt[:,0]<=w1)\
                               &(kp2_new_pt[:,1]<=h1)]
        # create cv.KeyPoint list
        kp2_new = [cv.KeyPoint(pt[0],pt[1],5) for pt in kp2_new_pt]
        
        # get probability map
        heatmap = self.probabilityMap(kp1,kp2_new,(w1,h1),100)
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        
        # combine heatmap with warped image
        display = cv.addWeighted(result,0.5,heatmap,0.3,0)
        
        # create a comparison image
        final = np.hstack((src,result,display))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(final,'reference',(0,25), font, 1,(0,255,0),2,cv.LINE_AA)
        cv.putText(final,'target',(w1,25), font, 1,(0,255,0),2,cv.LINE_AA)
        cv.putText(final,'comparison',(2*w1,25), font, 1,(0,255,0),2,cv.LINE_AA)
        
        # save the comparison
        cv.imwrite(dest,final)
