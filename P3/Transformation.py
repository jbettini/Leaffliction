import os
import os.path as osp
import matplotlib.pyplot as plt
import cv2
import argparse
import shutil
from plantcv import plantcv as pcv
import traceback
import numpy as np


def show_image_figure(images_to_display):
    size = len(images_to_display)
    if not images_to_display or size != 6:
        raise ValueError(f"Final Fig must contain only 6 image, actually {size}.")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for idx, tname in enumerate(images_to_display):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        ax.imshow(images_to_display[tname], cmap='gray')
        ax.axis('off')
        ax.set_title(tname)
    
    plt.tight_layout()
    plt.show()


#################################################################
# Transformation


final_fig = {}

def tresh_bin_mask(image, lower_thresh, upper_thresh):
    blurred_bgr = cv2.GaussianBlur(image, (5,5), 0)
    hsv_image = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
    binary_mask = cv2.inRange(hsv_image, np.array(lower_thresh), np.array(upper_thresh))
    return binary_mask

def _create_leaf_mask(image):
    lower_leaf = np.array([25, 0, 0])
    upper_leaf = np.array([95, 255, 255])
    mask = tresh_bin_mask(image, lower_leaf, upper_leaf)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# def _create_disease_mask(image):
#     lower_disease = np.array([10, 60, 50])
#     upper_disease = np.array([30, 255, 255])
    
#     disease_mask = tresh_bin_mask(image, lower_disease, upper_disease)
    
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
#     disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    
#     return disease_mask

# def draw_roi_disease(image):
# # Apple Scab
# # lower_thresh = np.array([35, 50, 50])
# # upper_thresh = np.array([100, 255, 255])
# # Apple Rot
# # lower_thresh = np.array([0, 20, 50])
# # upper_thresh = np.array([20, 255, 255])
# # Apple Rust + Rot
# # lower_thresh = np.array([0, 50, 50])
# # upper_thresh = np.array([25, 255, 255])
#     disease_mask = _create_disease_mask(image.copy())
#     contours, _ = cv2.findContours(disease_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     if contours:
#         cv2.drawContours(image, contours, -1, (0, 0, 255), -1)
#     return image


def original(image):
    return image


def mask(image):
    bin_mask = _create_leaf_mask(image)
    masked_image = pcv.apply_mask(img=image, mask=bin_mask, mask_color='white')
    return masked_image


def roi_objects(image):
    green_mask = _create_leaf_mask(image)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return image
    else:
        image_with_drawing = np.copy(image)
        # image_with_drawing = draw_roi_disease(image_with_drawing)
        cv2.drawContours(image_with_drawing, contours, -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        cv2.rectangle(image_with_drawing, (x, y), (x+w, y+h), (255, 0, 0), 3)
        return image_with_drawing
    

def draw_landmarks_circles(image, points, color):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)
        
def pseudolandmarks(image):
    pcv.params.sample_label = "plant"
    
    mask = _create_leaf_mask(image)
    
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img=image, mask=mask)
    left_landmarks = pcv.outputs.observations['plant']['left_lmk']['value']
    right_landmarks = pcv.outputs.observations['plant']['right_lmk']['value']
    center_landmarks = pcv.outputs.observations['plant']['center_h_lmk']['value']
    draw_landmarks_circles(image, left_landmarks, (0, 0, 255))
    draw_landmarks_circles(image, right_landmarks, (0, 255, 0))
    draw_landmarks_circles(image, center_landmarks, (255, 0, 0))
    
    return image
    
def analyse_obj(image):
    mask = _create_leaf_mask(image)
    
    rois = pcv.roi.auto_grid(mask=mask, nrows=2, ncols=2, radius=10, img=image)
    
    lbl_mask, n_lbls = pcv.create_labels(mask=mask, rois=rois)
    
    shape_img = pcv.analyze.size(img=image.copy(), labeled_mask=lbl_mask, n_labels=n_lbls, label="plant")
    return shape_img


def color_histogram(image):
    return image


tfonctions = {
    "--original": original,
    "--gaussian-blur": _create_leaf_mask,
    "--mask": mask,
    "--roi-objects": roi_objects,
    # "--analyse-object": analyse_obj,
    "--pseudolandmarks": pseudolandmarks,
}


# Transformation
#################################################################
# Utils

def is_valid_image_cv2(filepath):
    if not filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False


def get_new_filepath(output_dir, filename, ttype):
    base_name, extension = osp.splitext(filename)
    base_aug_filename = f"{base_name}_{ttype}{extension}"
    output_path = osp.join(output_dir, base_aug_filename)
    if not osp.exists(output_path):
        return output_path
    counter = 0
    while True:
        new_aug_filename = f"{base_name}_{ttype}_{counter}{extension}"
        new_output_path = osp.join(output_dir, new_aug_filename)
        if not osp.exists(new_output_path):
            return new_output_path
        counter += 1


def replace_root_dir(dirname, output_root):
    parts = osp.normpath(dirname).split(os.sep)

    if parts and parts[0] == '':
        parts = parts[1:]
    parts = parts[1:] if len(parts) > 1 else []

    new_path = osp.join(output_root, *parts)
    return new_path

def any_transformation_flag_set(flag_map):
    return any(flag_map.values())


def mask_is_not_set(tname, flag_map):
    if not any_transformation_flag_set(flag_map):
        return True
    return flag_map.get(tname, False)

# Utils
#################################################################


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Process images transformation and display an colors histogram"
        )
        parser.add_argument(
            "--file",
            help="Show file transformation"
        )
        parser.add_argument(
            "-src", "--source",
            help="Source Folder"
        )
        parser.add_argument(
            "-dst", "--destination",
            default="./transformed_directory",
            help="Destination Folder"
        )
        parser.add_argument(
            "-f", "--force",
            action="store_true",
            help="Force to overwrite the destination folder"
        )
        parser.add_argument(
            "--gaussian-blur",
            action="store_true",
            help="Apply gaussian-blur"
        )
        parser.add_argument(
            "--mask",
            action="store_true",
            help="Apply mask"
        )
        parser.add_argument(
            "--roi-objects",
            action="store_true",
            help="Apply roi-objects"
        )
        parser.add_argument(
            "--pseudolandmarks",
            action="store_true",
            help="Apply pseudolandmarks"
        )
        parser.add_argument(
            "--analyse-object",
            action="store_true",
            help="Apply analyse-object"
        )
        
        args = parser.parse_args()

        if args.file:
            if osp.isfile(args.file):
                if is_valid_image_cv2(args.file):
                    image = cv2.imread(args.file)
                    for tname, func in tfonctions.items():
                        timg = func(image.copy())
                        final_fig[tname[2::]] = cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)
                    show_image_figure(final_fig)
            else:
                parser.error(f"must be a correct file")
                exit(1)
                
        else:
            
            if not osp.exists(args.source):
                parser.error(f"Source path does not exist: {args.source}")
                exit(1)
            elif not osp.isdir(args.source):
                parser.error(f"Source path must be a directory: {args.source}")
                exit(1)
                
            if osp.exists(args.destination):
                if args.force:
                    shutil.rmtree(args.destination)
                else:
                    parser.error(f"{args.destination} exists, use --force to overwrite")
                    exit(1)
                    
                    
            shutil.copytree(args.source, args.destination)
            flag_map = {
                "--gaussian-blur": args.gaussian_blur,
                "--mask": args.mask,
                "--roi-objects": args.roi_objects,
                "--pseudolandmarks": args.pseudolandmarks,
                "--analyse-object": args.analyse_object
            }
            for root, dirs, files in os.walk(args.destination):
                if not dirs:
                    for entry in files:
                        filepath = osp.join(root, entry)                  
                        if is_valid_image_cv2(filepath):
                            image = cv2.imread(filepath)
                            for tname, func in tfonctions.items():
                                if mask_is_not_set(tname, flag_map):
                                    timg = func(image.copy())
                                    new_filepath = get_new_filepath(root, entry, tname[2::])
                                    cv2.imwrite(new_filepath, timg)

    except TypeError as e:
        print(f"TypeError: {str(e)}")
        traceback.print_exc()
    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
