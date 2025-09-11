import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
    try:
        if len(sys.argv) != 2:
            raise TypeError("Usage: python Distribution.py <arg1>: arg1 must be the folder path")
        path = sys.argv[1]
        folders = {}
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                files = [x for x in os.listdir(full_path)
                         if x.lower().endswith((('.jpg', '.jpeg', '.png')))]
                folders[entry] = len(files)
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