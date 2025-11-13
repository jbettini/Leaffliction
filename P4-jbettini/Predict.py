import os.path as osp
import tensorflow as tf
import argparse
import numpy as np


batch_size = 32
img_height = 255
img_width = 255

model_path = "leaffliction_model.keras"

class_names = [
    'Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
    'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot'
]


def model_summary(dataset, model):
    eval_ds = tf.keras.utils.image_dataset_from_directory(
        dataset,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    eval_ds = eval_ds.cache().shuffle(1000).prefetch(
                                                    buffer_size=AUTOTUNE)
    model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )
    score = model.evaluate(eval_ds, verbose=0)
    print(f"Model accuracy: {(score[1] * 100):.2f}%")


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Tain Model from a dataset"
        )

        parser.add_argument('path', help="Path to img file, or dataset")

        parser.add_argument(
            "--summary",
            action="store_true",
            help="Apply analyse-object"
        )

        path = parser.parse_args().path
        args = parser.parse_args()

        if not osp.exists(path):
            parser.error(f"Path does not exist: {path}.")
        elif not osp.exists(model_path):
            parser.error("Model not find.")

        model = tf.keras.models.load_model('leaffliction_model.keras')

        if args.summary:
            if not osp.isdir(path):
                raise TypeError("Dataset must be a directory")
            model_summary(path, model)
        else:
            if not osp.isfile(path):
                raise TypeError("Path must be a file if flag summary not use")
            img = tf.keras.utils.load_img(args.path, target_size=(255, 255))

            img_array = tf.keras.utils.img_to_array(img)
            img_batch = tf.expand_dims(img_array, 0)

            print("Prédiction ...")

            predictions = model.predict(img_batch)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            print(f"Prediction : '{predicted_class}'")

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
