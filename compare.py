import argparse
import os

from src import ImageComparator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="Reference image name")
    parser.add_argument("tar", help="Target image name")
    parser.add_argument("--diff", help="Compare images with raw difference", action="store_true")
    parser.add_argument("--bgs", help="Compare images with background subtraction", action="store_true")
    parser.add_argument("--kp", help="Compare images by comparing keypoints", action="store_true")
    args = parser.parse_args()
    
    ref_name = os.path.splitext(args.ref)[0]
    tar_name = os.path.splitext(args.tar)[0]
    dest = ref_name + "_" + tar_name + ".jpg"
    
    method = "KP" if args.kp \
             else "BGS" if args.bgs \
             else "diff"

    comparator = ImageComparator()
    comparator.compare(os.path.join('images/',args.ref),\
                       os.path.join('images/',args.tar),\
                       os.path.join('comparisons/',dest),\
                       method)
