import os
import os.path as osp
import argparse
import shutil
import random

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Create a test data set by moving files from a source directory."
        )
        parser.add_argument('path', help="Source dataset path")
        data_dir = parser.parse_args().path

        if not osp.exists(data_dir):
            parser.error(f"Dataset path does not exist: {data_dir}.")
        elif not osp.isdir(data_dir):
            parser.error("Dataset must be a folder.")

        test_directory_name = 'test_directory'
        num_files_to_move = 20
        
        os.makedirs(test_directory_name, exist_ok=True)

        for current_folder, subfolders, files in os.walk(data_dir):
            if not files:
                continue

            if len(files) < num_files_to_move:
                files_to_select = files
            else:
                random.shuffle(files)
                files_to_select = files[:num_files_to_move]

            destination_folder = current_folder.replace(data_dir, test_directory_name, 1)
            os.makedirs(destination_folder, exist_ok=True)

            for file_name in files_to_select:
                source_path = os.path.join(current_folder, file_name)
                destination_path = os.path.join(destination_folder, file_name)
                shutil.move(source_path, destination_path)

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")

if __name__ == "__main__":
    main()