# Chest X-Ray Disease Classifier

A web application that leverages deep learning to classify chest X-ray images for pneumonia and other diseases. The solution features a backend API for predictions, an intuitive frontend for image upload and results display, and tools for load testing and performance analysis.

The machine learning core is a convolutional neural network (CNN) trained on labeled chest X-ray datasets, enabling it to distinguish between normal and diseased images with high accuracy. The model employs data augmentation, dropout, batch normalization, and early stopping to boost generalization and prevent overfitting. 

The API (built in FastAPI or Flask) allows users to submit single or batch images and receive diagnostic predictions with confidence scores. The frontend offers a simple interface for non-technical users to interact with the system.

To ensure robustness and scalability, Locust is used for load testing, analyzing throughput and latency under concurrent use. This project demonstrates the practical application of AI in medical imaging, providing a modular, reliable, and accessible diagnostic aid.

---

## 📺 Demo Video

Watch the full demo on YouTube:  
https://youtu.be/Lpywqb8_IXo

---

## 🌍 Live Demo

[https://chest-xray-classifier-9tuew99kyhj2xzbhbaix9r.streamlit.app/](https://chest-xray-classifier-9tuew99kyhj2xzbhbaix9r.streamlit.app/)

---

## 📝 Project Description

This project provides an end-to-end pipeline for automated chest X-ray disease classification, featuring:
- A convolutional neural network (CNN) trained on chest X-ray images
- A REST API for inference using FastAPI
- Batch and single-image prediction endpoints
- Load testing using Locust
- Example code for data loading, training, and prediction

---

## 🚀 Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/O-keita/chest-xray-classifier.git
    cd chest-xray-classifier
    ```

2. **(Optional) Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the API locally:**
    ```bash
    uvicorn src.api:app --reload
    # or
    python src/api.py
    ```

5. **Test the API:**
    - Visit [https://chest-xray-classifier-1.onrender.com](https://chest-xray-classifier-1.onrender.com/docs) to see the API documentation.
    - Use `/predict` to upload and classify images.

6. **Load Testing with Locust:**
    ```bash
    locust -f locustfile.py --host=https://chest-xray-classifier-1.onrender.com
    # Open https://chest-xray-classifier-1.onrender.com to configure and start your test
    ```

---

## 📊 Model Accuracy and Regularization (Best Epoch)

**Best Epoch:** 5

At epoch 5, the model achieved its highest validation accuracy and AUC:

| Metric          | Training | Validation |
|-----------------|----------|------------|
| Accuracy        | 0.903    | 0.875      |
| AUC             | 0.968    | 0.984      |
| Loss            | 0.252    | 0.260      |
| Precision       | 0.903    | 0.875      |
| Recall          | 0.903    | 0.875      |

### 🛡️ Regularization & Generalization Techniques Used

- **Data Augmentation:**  
  The training images were randomly rotated, shifted, and horizontally flipped to increase data diversity and improve generalization.

- **Dropout Layer:**  
  Dropout with a rate of 0.5 was applied to the dense layer, randomly disabling 50% of neurons during training to prevent overfitting.

- **Batch Normalization:**  
  Batch normalization layers were included after each convolutional layer to stabilize and accelerate training, and to provide a regularization effect.

- **Early Stopping:**  
  Training was monitored using early stopping with patience of 5 epochs. If the validation loss did not improve for 5 consecutive epochs, training was halted and the best model (with lowest validation loss) was restored.

- **Model Checkpoint:**  
  The model was automatically saved at its best validation loss during training, ensuring only the best-performing parameters are used.

---

**Summary:**  
At the fifth epoch, the model demonstrated excellent generalization with 87.5% validation accuracy and 98.4% validation AUC, thanks to robust regularization and early stopping strategies.

---

## 🧠 Techniques Used

- **Data Augmentation:** Random rotations, shifts, and flips for better generalization.
- **CNN Architecture:** Stacked Conv2D layers with batch normalization, max pooling, dropout, and dense layers.
- **Regularization:** Dropout layers, early stopping, and model checkpointing to prevent overfitting.
- **Evaluation Metrics:** Accuracy, AUC, precision, and recall for comprehensive performance tracking.
- **Batch and Single Prediction:** Support for single image and batch predictions.
- **Load Testing:** Locust used to simulate concurrent users and measure API robustness.

---

## 🌊 Flood Request Simulation Results

<img width="1217" height="900" alt="total_requests_per_second_1754183442 268" src="https://github.com/user-attachments/assets/46c47296-e8b3-43f8-8d44-76c0e1936da0" />

---

## 📎 Useful Links

- [YouTube Demo](YOUR_YOUTUBE_DEMO_LINK_HERE)
- [Live Demo](https://chest-xray-classifier-9tuew99kyhj2xzbhbaix9r.streamlit.app/)
- [API Docs](https://chest-xray-classifier-1.onrender.com/docs)

---

## 🛠️ Technologies Used

- Python, TensorFlow/Keras, FastAPI, Locust

