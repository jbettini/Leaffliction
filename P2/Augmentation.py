import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import argparse
import shutil

#
# -- show
# -- balance
# -- dir
# -- file
# 

######################################################
# Show arg

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
    
######################################################
########################
def is_valid_image_cv2(filepath):
    if not filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False

def append_ag_type(filename, ag):
    name, ext = os.path.splitext(filename)
    return f"{name}_{ag}{ext}"

def augmentation(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = {
        # 70% of chance to apply an effect
        'Flip': A.Compose([
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
        ]),
        # Rotate image between -90 and 90 degrees
        'Rotate': A.SafeRotate(limit=90, p=1.0),
        'HighContrast': A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        'Brightness': A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), p=1.0),
        # 50% of chance to apply different blur
        'Blur': A.OneOf([
            A.GaussianBlur(blur_limit=(11, 17), p=1.0),
            A.MotionBlur(blur_limit=(11, 17), p=1.0),
        ], p=1.0),
        # Divide Img in 5x5 grid and apply a Distortion
        'Distortion': A.GridDistortion(num_steps=10, distort_limit=0.8, p=1.0),
    }
    res = {}
    
    for tname, t in transforms.items():
        result = t(image=image)
        res[tname] = result['image']
    res['Original'] = image
    return res
                
def create_augmented_parent_dir(filepath, output_dir):
    parts = filepath.split(os.sep)
    if len(parts) < 3:
        raise ValueError(
            "Error: Dataset must have folder depth of 3 like:\n"
            "Dataset -> Subset1 -> imgfiles"
        )
    last_parts = parts[-3:]
    sub_path = os.path.join(*last_parts)
    target_subdirectory = os.path.dirname(sub_path)
    final_output_dir = os.path.join(output_dir, target_subdirectory)
    os.makedirs(final_output_dir, exist_ok=True)
    return final_output_dir



def handle_file(filepath, args):
    if is_valid_image_cv2(filepath):
        ag_path = create_augmented_parent_dir(filepath, args.output)
        ag_imgs = augmentation(filepath)
        if not ag_imgs:
            raise ValueError("Error: imgs not augmented.")
        
        if len(final_fig) <= 6:
            final_fig.append(ag_imgs)

        filename = os.path.basename(filepath)
        for key, value in ag_imgs.items():
            ag_filename = append_ag_type(filename, key)
            output_path = os.path.join(ag_path, ag_filename)
            if not os.path.exists(output_path):
                brg_img = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, brg_img)
        return True
    return False

def main():
    try:
        
        parser = argparse.ArgumentParser(description ="Process balance and image augmentation in a dataset")
        parser.add_argument('path', help ="An path to a dir or file")
        parser.add_argument("-o", "--output", default="./augmented_directory" ,help ="Output Folder")
        parser.add_argument("--show", action="store_true" ,help ="Show between 1 and 6 images augmented")
        parser.add_argument("--balance", action="store_true" ,help ="Balance a dataset directory")
        parser.add_argument("-f", "--force", action="store_true" ,help ="Force to overwrite the output folder")
        args = parser.parse_args()
        
        if not os.path.exists(args.path):
            parser.error(f"Source path does not exist: {args.path}")

        if args.balance and os.path.isfile(args.path):
            parser.error("--balance mode can only be used with a directory.")
                
        if os.path.exists(args.output):
            if args.force:
                shutil.rmtree(args.output)
            else:
                parser.error(f"{args.output} already present use --force flag to overwrite/delete")
            
        if args.balance:
            print("Balance not handled")
        else:
            if os.path.isfile(args.path):
                handle_file(args.path, args)
            else:
                print("Augmenting directory without balencing.")
                processed_file = 0
                for entry in os.listdir(args.path):
                    full_path = os.path.join(args.path, entry)
                    if os.path.isdir(full_path):
                        for img in os.listdir(full_path):
                            if handle_file(os.path.join(full_path, img), args):
                                processed_file += 1
                if processed_file == 0:
                    print("No valid img file found.")
                            
        if args.show:
            create_final_figure(final_fig)

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()