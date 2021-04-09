'''
Full pipeline for comparing images
'''

import numpy as np
import cv2 as cv
from imgmatching import ImageMatcher, ImageMatcherSG

class Comparator():
    def __init__(self,imgmatcher,imgcomparer):
        '''
        Parameters
        ----------
        imgmatcher : ImageMatcher
        imgcomparer : ImageComparer
        '''
        self.imgmatcher = imgmatcher
        self.imgcomparer = imgcomparer

    def compare(self,src,tar,dest):
        '''
        Parameters
        ----------
        src : string
            path to the reference image
        tar : string
            path to the target image to be compared with the reference
        dest : string
            path where to save the comparison
        '''
        src = cv.imread(src)
        tar = cv.imread(tar)

        w1, h1 = src.shape[1], src.shape[0]
        w2, h2 = tar.shape[1], tar.shape[0]

        ### Computation part

        src, res, kp = self.imgmatcher.match(src,tar)
        heatmap = self.imgcomparer.compare(src,res)

        ### Display part

        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

        # Combine heatmap with warped image
        comparison = cv.addWeighted(res,0.5,heatmap,0.3,0)

        # Create a comparison image
        final = np.hstack((src,res,comparison))

        # Add text
        font = cv.FONT_HERSHEY_SIMPLEX
        color = (255,255,0)
        cv.putText(final,'reference',(25,25), font, 1,color,2,cv.LINE_AA)
        cv.putText(final,'warped target',(w1+25,25), font, 1,color,2,cv.LINE_AA)
        cv.putText(final,'comparison',(2*w1+25,25), font, 1,color,2,cv.LINE_AA)

        # Save the comparison
        cv.imwrite(dest,final)

if __name__ == '__main__':
    from imgcomparing import *
    cmp = Comparator(ImageMatcherSG(),PatchDiffCMP(11,5,5,gaussian=True))
    cmp.compare('images/ref.jpg','images/tar.jpg','comp.jpg')