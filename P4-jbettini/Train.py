import os
import os.path as osp
import tensorflow as tf
import argparse


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 32
img_height = 255
img_width = 255


subsets = [
    'Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
    'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot'
]


def save_dataset_as_images(dataset, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name in class_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    file_counter = 0
    for image_batch, label_batch in dataset:
        for i in range(len(image_batch)):
            image = image_batch[i]
            label_index = label_batch[i]
            class_name = class_names[label_index]
            save_path = os.path.join(
                                output_dir,
                                class_name,
                                f"image_{file_counter}.png"
                        )
            tf.keras.utils.save_img(save_path, image)
            file_counter += 1


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Tain Model from a dataset"
        )

        parser.add_argument('path', help="Dataset Path")

        data_dir = parser.parse_args().path

        if not osp.exists(data_dir):
            parser.error(f"Dataset path does not exist: {data_dir}.")
        elif not osp.isdir(data_dir):
            parser.error("Dataset must be a folder.")

        # Split init datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        print("Saving Datasets ...")
        save_dataset_as_images(train_ds, subsets, "training_dataset")
        save_dataset_as_images(val_ds, subsets, "validation_dataset")

        class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(
                                                        buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        num_classes = len(class_names)

        model = Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(
                        optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        metrics=['accuracy']
                    )

        epochs = 10

        _ = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        model.save('leaffliction_model.keras')

    except TypeError as e:
        print(f"TypeError: {str(e)}")
    except BaseException as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
