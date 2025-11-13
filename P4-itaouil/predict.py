import argparse
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_class_names(model_dir):
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    if not os.path.exists(class_names_path):
        msg = "Class names file not found: {}".format(class_names_path)
        raise FileNotFoundError(msg)

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def predict_image(image_path, model_path, class_names,
                  img_height=224, img_width=224, show_plot=True):
    model = keras.models.load_model(model_path)

    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    predicted_class = class_names[predicted_class_idx]

    print("\n{}".format('='*50))
    print("Image: {}".format(image_path))
    print("Predicted disease: {}".format(predicted_class))
    print("Confidence: {:.2f}%".format(confidence*100))
    print("{}\n".format('='*50))

    print("All class probabilities:")
    sorted_indices = np.argsort(predictions[0])[::-1]
    for idx in sorted_indices:
        prob = predictions[0][idx]*100
        print("  {}: {:.2f}%".format(class_names[idx], prob))

    if show_plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        img_display = image.load_img(image_path)
        plt.imshow(img_display)
        title = "Predicted: {}\nConfidence: {:.1f}%".format(
            predicted_class, confidence*100)
        plt.title(title)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, predictions[0] * 100)
        plt.yticks(y_pos, class_names)
        plt.xlabel('Confidence (%)')
        plt.title('Prediction Probabilities')
        plt.xlim(0, 100)

        for i, v in enumerate(predictions[0] * 100):
            plt.text(v + 1, i, '{:.1f}%'.format(v), va='center')

        plt.tight_layout()
        plt.show()

    return predicted_class, confidence, predictions[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Predict plant disease from an image using '
                    'trained model'
    )
    parser.add_argument(
        'image',
        help='Path to the image file to classify'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable visualization plot'
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Error: Image file not found: {}".format(args.image))
        exit(1)

    model_path = './model/best_model.keras'
    if not os.path.exists(model_path):
        print("Error: Model file not found: {}".format(model_path))
        exit(1)

    class_names = load_class_names('./model')

    predict_image(
        args.image,
        model_path,
        class_names,
        show_plot=not args.no_plot
    )
