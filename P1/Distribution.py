import os
import sys
import matplotlib.pyplot as plt
import cv2


def is_valid_image_cv2(filepath):
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False


def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python script.py <path>")
            sys.exit(1)
        path = sys.argv[1]

        if not os.path.isdir(path):
            raise TypeError(f"Error: Path is not a directory: {path}")

        folders = {}
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                files = [f for f in os.listdir(full_path)
                         if is_valid_image_cv2(os.path.join(full_path, f))]
                folder_name = os.path.basename(full_path)
                folders[folder_name] = len(files)

        if not folders or sum(folders.values()) == 0:
            print("\nWarning: No valid images found. Nothing to plot.")
            return

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.pie(folders.values(), labels=folders.keys(), autopct='%1.1f%%')
        plt.subplot(1, 2, 2)
        plt.bar(folders.keys(), folders.values())
        plt.tight_layout()
        plt.show()
    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
