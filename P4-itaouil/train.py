import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def create_model(num_classes, img_height=224, img_width=224):
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def plot_training_history(history, output_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    msg = os.path.join(output_dir, 'training_history.png')
    print("Training history plot saved to " + msg)


def train_model(data_dir, output_dir, epochs=10,
                img_height=224, img_width=224):
    os.makedirs(output_dir, exist_ok=True)

    preprocess = keras.applications.mobilenet_v2.preprocess_input
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print("\nFound {} classes:".format(num_classes))
    for class_name, class_idx in train_generator.class_indices.items():
        print("  {}: {}".format(class_idx, class_name))

    class_names_path = os.path.join(output_dir, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        items = train_generator.class_indices.items()
        for class_name, class_idx in sorted(items, key=lambda x: x[1]):
            f.write("{}\n".format(class_name))

    model = create_model(num_classes, img_height, img_width)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    class StopAt90Percent(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy') > 0.92:
                print("\nAccuracy above 92% reached! Stopping training.")
                self.model.stop_training = True

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        # StopAt90Percent()
    ]

    print("\nStarting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)

    plot_training_history(history, output_dir)

    val_loss, val_accuracy = model.evaluate(
        validation_generator, verbose=0
    )
    print("\nTraining complete!")
    print("Final accuracy: {:.2f}%".format(val_accuracy*100))
    print("Models saved to {}/".format(output_dir))

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train a deep learning model for plant disease '
                    'classification'
    )
    parser.add_argument(
        '-data',
        '--data_dir',
        default='./images',
        help='Directory containing subdirectories of images '
             '(default: ./images)'
    )
    parser.add_argument(
        '-out',
        '--output_dir',
        default='./model',
        help='Directory to save the trained model (default: ./model)'
    )

    args = parser.parse_args()

    train_model(
        args.data_dir,
        args.output_dir
    )
