import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import albumentations as A
import cv2

#
# -- show
# -- balance
# -- dir
# -- file
# 

final_fig = []

def create_final_figure(images_to_display):
    if not images_to_display:
        return

    aug_names = list(images_to_display[0].keys())
    num_rows = len(images_to_display)
    num_cols = len(aug_names)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3), squeeze=False)

    for row_idx, image_dict in enumerate(images_to_display):
        for col_idx, aug_name in enumerate(aug_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(image_dict[aug_name])
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(aug_name)
    
    plt.tight_layout()
    plt.show()

def is_valid_image_cv2(filepath):
    if not filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False

def create_augmented_parent_dir(filepath):
    if not os.path.exists("./augmented_directory"):
        os.mkdir("./augmented_directory")
    directory = os.path.dirname(filepath)
    parts = filepath.split(os.sep)
    if parts[0] == "images" or parts[0] == "augmented_directory":
        relative_parts = parts[1:]
    else:
        relative_parts = parts
        
    if len(relative_parts) != 3:
        raise TypeError(f"Error: img file must have a folder depth of 3 after images, len : {len(parts)} for {filepath} ")
    relative_path = os.path.join(*relative_parts)
    directory = os.path.dirname(relative_path)
    os.makedirs(f"./augmented_directory/{directory}", exist_ok=True)
    return f"./augmented_directory/{directory}"
 
def augmentation(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = {
        'Flip': A.HorizontalFlip(p=1.0),
        'Rotate': A.Rotate(limit=60, p=1.0),
        'HighContrast': A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        'Brightness': A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), p=1.0),
        'Blur': A.MedianBlur(blur_limit=(11, 17), p=1.0), 
        'Zoom': A.Affine(scale=(1.2, 1.5), p=1.0),
    }
    res = {}
    
    for tname, t in transforms.items():
        result = t(image=image)
        res[tname] = result['image']
    res['Original'] = image
    return res

def append_ag_type(filename, ag):
    name, ext = os.path.splitext(filename)
    return f"{name}_{ag}{ext}"

def handle_file(filepath):
    if is_valid_image_cv2(filepath):
        ag_dir = create_augmented_parent_dir(filepath)
        ag_img = augmentation(filepath)
        if len(final_fig) <= 6:
            final_fig.append(ag_img)
        filename = os.path.basename(filepath)
        for key, value in ag_img.items():
            ag_filename = append_ag_type(filename, key)
            brg_img = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(ag_dir, ag_filename)
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, brg_img)

def handle_dataset(setpath):
    for entry in os.listdir(setpath):
        full_path = os.path.join(setpath, entry)
        if os.path.isdir(full_path):
            for img in os.listdir(full_path):
                handle_file(os.path.join(full_path, img))
    
def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Usage: python Augmentation.py <arg1>: arg1 must be the img to transform or the directory")
        path = sys.argv[1]
        if not os.path.exists(path):
            raise TypeError("Path does not exist")
        if os.path.isdir(path):
            handle_dataset(path)
        else:
            handle_file(path)
        create_final_figure(final_fig)
    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()