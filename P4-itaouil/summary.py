import os
import os.path as osp
import tensorflow as tf
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
MODEL_PATH = "model/best_model.keras"
VALIDATION_SPLIT = 0.2
DATA_SEED = 123


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Evaluate a Keras model on its validation dataset"
        )
        parser.add_argument(
            'dataset_path',
            help="Path to the *root* dataset folder"
        )

        args = parser.parse_args()
        dataset_path = args.dataset_path

        if not osp.exists(dataset_path):
            parser.error(f"Dataset path does not exist: {dataset_path}")
        if not osp.isdir(dataset_path):
            parser.error("The dataset path must be a directory.")
        if not osp.exists(MODEL_PATH):
            parser.error(f"Model not found at: {MODEL_PATH}")

        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)

        print("Compiling model...")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False
            ),
            metrics=['accuracy']
        )

        print(f"Loading validation dataset from {dataset_path}...")
        eval_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=DATA_SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        class_names = eval_ds.class_names
        print(f"Found {len(class_names)} classes: {class_names}")

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        eval_ds = eval_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE
        eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)

        print("Evaluating model performance on validation set...")
        results = model.evaluate(eval_ds)

        print("\n--- Evaluation Complete ---")
        print(f"Test Loss:     {results[0]:.4f}")
        print(f"Test Accuracy: {results[1] * 100:.2f}%")

    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    main()
