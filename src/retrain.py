import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Ensure eager execution is on

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from preprocessing import load_data

MODEL_PATH = "models/best_model.h5"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 2

def retrain_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Recompile with a new optimizer instance!
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )

    print("Loading data...")
    train_gen, val_gen, test_gen, class_indices = load_data(
        TRAIN_DIR, VAL_DIR, TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    )
    print("Retraining...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    print("Saving model...")
    model.save(MODEL_PATH)
    print("Retraining complete.")

if __name__ == "__main__":
    retrain_model()