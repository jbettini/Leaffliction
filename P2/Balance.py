import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import albumentations as A
import cv2

def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Usage: python Distribution.py <arg1>: arg1 must be the img to transform or the directory")
        path = sys.argv[1]
        if not os.path.exists(path):
            raise TypeError("Path does not exist")
    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()