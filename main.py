import argparse
import os

from comparator import *
from imgcomparing import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="Reference image image")
    parser.add_argument("tar", help="Target image image")
    parser.add_argument("--diff", help="Compare images with raw difference", action="store_true")
    parser.add_argument("--bgs", help="Compare images with background subtraction", action="store_true")
    parser.add_argument("--flow", help="Compare images with optical flow", action="store_true")
    args = parser.parse_args()
    
    ref_name = os.path.splitext(args.ref)[0]
    tar_name = os.path.splitext(args.tar)[0]
    dest = ref_name + "_" + tar_name + ".jpg"
    
    method = BGSubCMP() if args.bgs \
             else OptFlowCMP() if args.flow \
             else RawDiffCMP()

    comparator = Comparator(method)
    comparator.compare(os.path.join('images/',args.ref),\
                       os.path.join('images/',args.tar),\
                       os.path.join('comparisons/',dest))
