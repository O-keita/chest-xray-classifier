import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

def process_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model, img_path, class_labels, target_size=(128, 128)):
    img_array = process_image(img_path, target_size)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    return class_labels[predicted_class_index], confidence

def predict_batch(model, img_paths, class_labels, target_size=(128, 128)):
    results = []
    for img_path in img_paths:
        try:
            label, confidence = predict_image(model, img_path, class_labels, target_size)
            results.append({'image': img_path, 'label': label, 'confidence': confidence})
        except Exception as e:
            results.append({'image': img_path, 'error': str(e)})
    return results

if __name__ == "__main__":
    model_path = 'models/best_model.h5'
    img_paths = ['data/train/PNEUMONIA/person3_bacteria_10.jpeg', 'data/train/NORMAL/IM-0115-0001.jpeg']
    class_labels = {0: 'Normal', 1: 'Pneumonia'}  # Corrected spelling

    model = load_trained_model(model_path)
    results = predict_batch(model, img_paths, class_labels)
    for result in results:
        if 'label' in result:
            print(f"Predicted label for {result['image']}: {result['label']} (confidence: {result['confidence']:.2f})")
        else:
            print(f"Error predicting {result['image']}: {result['error']}")