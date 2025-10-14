import os
import os.path as osp
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import argparse
import shutil
import random
from pathlib import Path


final_fig = []
thresh = 600


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


def augmentation(filepath, num_of_Aug):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_transforms = {
        # 70% of chance to apply an effect
        'Flip': A.Compose([
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
        ]),
        # Rotate image between -90 and 90 degrees
        'Rotate': A.SafeRotate(limit=90, p=1.0),
        'HighContrast': A.CLAHE(clip_limit=(1.0, 3.0), p=1.0),
        'Brightness': A.RandomBrightnessContrast(
            brightness_limit=(0.3, 0.3), p=1.0
        ),

        # Erase some part of the image to hide some parts of the image
        'RandomErasing': A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(40, 80),
            hole_width_range=(40, 80),
            fill=0,
            p=1.0
        ),
        # Divide Img in 10*10 grid and apply a Distortion
        'Distortion': A.GridDistortion(num_steps=10, distort_limit=0.4, p=1.0),
    }
    res = {}
    items_as_list = list(all_transforms.items())
    selected_transformation = random.sample(items_as_list, num_of_Aug)
    for tname, t in selected_transformation:
        result = t(image=image)
        res[tname] = result['image']
    # res['Original'] = image
    return res


def handle_file(filepath, num_of_Aug=6):
    if is_valid_image_cv2(filepath):
        ag_imgs = augmentation(filepath, num_of_Aug)
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


def balance_dir(dirpath):
    sets_data = {}
    for root, dirs, files in os.walk(dirpath):
        valid_images = [f for f in files
                        if is_valid_image_cv2(osp.join(root, f))]
        if valid_images and not dirs:
            set_name = Path(root).parent.name
            if set_name not in sets_data:
                sets_data[set_name] = {'max_files': 0, 'folders': []}
            nimg = len(valid_images)
            current_max = sets_data[set_name]['max_files']
            sets_data[set_name]['max_files'] = max(current_max, nimg)
            sets_data[set_name]['folders'].append((root, nimg))

    for set_name, data_map in sets_data.items():
        max_aug = thresh
        cur_max_files = data_map['max_files']
        if cur_max_files == 0:
            continue
        elif cur_max_files > max_aug:
            max_aug = cur_max_files
        for subset_path, num_of_file in data_map['folders']:
            if num_of_file >= max_aug:
                continue
            source_images = [f for f in os.listdir(subset_path)
                             if is_valid_image_cv2(osp.join(subset_path, f))]
            if not source_images:
                continue
            iter = 0
            images_needed = max_aug - num_of_file
            num_of_iter = images_needed // 6
            while iter < num_of_iter:
                image_to_augment = random.choice(source_images)
                full_image_path = osp.join(subset_path, image_to_augment)
                handle_file(full_image_path)
                iter += 1
            remaining_img = images_needed % 6
            if remaining_img:
                image_to_augment = random.choice(source_images)
                full_image_path = osp.join(subset_path, image_to_augment)
                handle_file(full_image_path, remaining_img)


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Process balance and image augmentation in a dataset"
        )
        parser.add_argument('path', help="An path to a dir or file")
        parser.add_argument(
            "-o", "--output",
            default="./augmented_directory",
            help="Output Folder"
        )
        parser.add_argument(
            "--show",
            action="store_true",
            help="Show between 1 and 6 images augmented"
        )
        parser.add_argument(
            "--balance",
            action="store_true",
            help="Balance a dataset directory"
        )
        parser.add_argument(
            "-f", "--force",
            action="store_true",
            help="Force to overwrite the output folder"
        )
        args = parser.parse_args()
        if not osp.exists(args.path):
            parser.error(f"Source path does not exist: {args.path}")

        if args.balance and osp.isfile(args.path):
            parser.error("--balance mode can only be used with a directory.")

        if osp.exists(args.output):
            if args.force:
                shutil.rmtree(args.output)
            else:
                parser.error(f"{args.output} exists, use --force to overwrite")

        if osp.isfile(args.path):
            output_tree = replace_root_dir(osp.dirname(args.path), args.output)
            output_file = osp.join(output_tree, osp.basename(args.path))
            os.makedirs(osp.dirname(output_file), exist_ok=True)
            shutil.copy2(args.path, output_file)
            handle_file(output_file)
        else:
            shutil.copytree(args.path, args.output)
            if args.balance:
                print("Balancing directory...")
                balance_dir(args.output)
            else:
                processed_file = 0
                print("Augmenting directory without balencing...")
                for root, _, files in os.walk(args.output):
                    for entry in files:
                        handle_file(osp.join(root, entry))
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
