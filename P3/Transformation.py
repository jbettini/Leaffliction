import os
import os.path as osp
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import argparse
import shutil
from plantcv import plantcv as pcv


final_fig = []


def create_final_figure(images_to_display):
    if not images_to_display:
        return

    aug_names = list(images_to_display[0].keys())
    num_rows = len(images_to_display)
    num_cols = len(aug_names)

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(num_cols * 3, num_rows * 3),
        squeeze=False
    )

    for row_idx, image_dict in enumerate(images_to_display):
        for col_idx, aug_name in enumerate(aug_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(image_dict[aug_name])
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(aug_name)

    plt.tight_layout()
    plt.show()


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


def get_new_filepath(output_dir, filename, aug_type):
    base_name, extension = osp.splitext(filename)
    base_aug_filename = f"{base_name}_{aug_type}{extension}"
    output_path = osp.join(output_dir, base_aug_filename)
    if not osp.exists(output_path):
        return output_path
    counter = 0
    while True:
        new_aug_filename = f"{base_name}_{aug_type}_{counter}{extension}"
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

# Utils
#################################################################

def transformation():
    pass

def handle_file(filepath, num_of_Aug=6):
    if is_valid_image_cv2(filepath):
        ag_imgs = transformation(filepath, num_of_Aug)
        if not ag_imgs:
            raise ValueError("Error: imgs not augmented.")

        if len(final_fig) <= 6:
            final_fig.append(ag_imgs)

        output_dir = osp.dirname(filepath)
        original_filename = osp.basename(filepath)
        for key, value in ag_imgs.items():
            new_file = get_new_filepath(output_dir, original_filename, key)
            brg_img = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_file, brg_img)
        return True
    return False

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Process images transformation and display an colors histogram"
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
            "--show",
            action="store_true",
            help="Show between 1 and 6 images augmented"
        )

        parser.add_argument(
            "-f", "--force",
            action="store_true",
            help="Force to overwrite the destination folder"
        )
        
        args = parser.parse_args()
        if not osp.exists(args.source):
            parser.error(f"Source path does not exist: {args.path}")

        if osp.exists(args.destination):
            if args.force:
                shutil.rmtree(args.destination)
            else:
                parser.error(f"{args.destination} exists, use --force to overwrite")

        if osp.isfile(args.path):
            images = handle_file(args.source)
        else:
            shutil.copytree(args.path, args.destination)
            for root, _, files in os.walk(args.destination):
                for entry in files:
                    handle_file(osp.join(root, entry))
        if args.show:
            create_final_figure(final_fig)

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
