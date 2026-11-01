"""Optional - CNN classifier with accuracy-vs-epochs plot (report section 4.3 / 5.1).

    python train_cnn.py            (needs: pip install tensorflow)

Conv2D/MaxPool x3 -> Flatten -> Dense, 128x128 input, batch 32, 10 epochs,
80/20 train/validation split. Use it in recognition with:  recognize.py --backend cnn
"""
import json

import config


def main():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    common = dict(validation_split=config.CNN_VAL_SPLIT, seed=42, image_size=config.CNN_IMG_SIZE,
                  batch_size=config.CNN_BATCH_SIZE, label_mode="categorical")
    train = tf.keras.utils.image_dataset_from_directory(config.DATASET_DIR, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(config.DATASET_DIR, subset="validation", **common)
    classes = train.class_names
    if len(classes) < 2:
        raise SystemExit("Need at least 2 people in the dataset to train a classifier.")

    model = models.Sequential([
        layers.Input(shape=(*config.CNN_IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomRotation(0.03), layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1), layers.RandomFlip("horizontal"),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.5),
        layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train, validation_data=val, epochs=config.CNN_EPOCHS)

    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    model.save(config.CNN_MODEL_FILE)
    config.CNN_CLASSES_FILE.write_text(json.dumps(classes))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="validation")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Accuracy per epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(config.OUTPUT_DIR / "accuracy_vs_epochs.png", dpi=150)
    print(f"Final validation accuracy: {hist.history['val_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
