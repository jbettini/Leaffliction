import os.path as osp
import tensorflow as tf
import argparse

class_names = [
    'Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
    'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot'
]


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Tain Model from a dataset"
        )

        parser.add_argument('path', help="Path to img file")

        img_path = parser.parse_args().path

        if not osp.exists(img_path):
            parser.error(f"Dataset path does not exist: {img_path}.")

        args = parser.parse_args()

        model = tf.keras.models.load_model('leaffliction_model.keras')

        img = tf.keras.utils.load_img(args.path, target_size=(255, 255))

        img_array = tf.keras.utils.img_to_array(img)
        img_batch = tf.expand_dims(img_array, 0)

        print("Prédiction en cours...")
        predictions = model.predict(img_batch)

        # score = tf.nn.softmax(predictions[0])

        # predicted_class = class_names[np.argmax(score)]
        # rate = 100 * np.max(score)
        # print(f"Cette image appartient très probablement à la classe : '{predicted_class}'")
        # print(f"Confiance : {rate:.2f}%")

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
