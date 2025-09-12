import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import Augmentation
import random


def is_valid_image_cv2(filepath):
    if not filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False

def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Usage: python Distribution.py <arg1>: arg1 must be the folder path")
        path = sys.argv[1]
        
        if not os.path.isdir(path):
            raise TypeError(f"Error: The provided path is not a directory: {path}")
        parts = path.split(os.sep)
        if not parts[0] == "augmented_directory":
            raise TypeError(f"Error: Use Augmented directory generated with Augmentation.py script")
        folders = {}
        max = 0
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                files = [f for f in os.listdir(full_path)
                         if is_valid_image_cv2(os.path.join(full_path, f))]
                if max < len(files):
                    max = len(files)
                if not len(files) == 0:
                    folders[full_path] = len(files)
                    
        if not folders or sum(folders.values()) == 0:
            print("\nWarning: No valid images were found in any subdirectories. No dataset to balance.")
            return

        for folder, count in folders.items():
            if count >= max:
                continue
            source_images = [f for f in os.listdir(folder) if is_valid_image_cv2(os.path.join(folder, f))]
            if not source_images:
                continue
            
            images_needed = max - count

            generated_count = 0
            while generated_count < images_needed:
                image_to_augment = random.choice(source_images)
                full_image_path = os.path.join(folder, image_to_augment)
                Augmentation.handle_file(full_image_path) 
                generated_count += 6

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()