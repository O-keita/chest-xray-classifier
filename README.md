# Chest X-Ray Disease Classifier

A web application that uses deep learning to classify chest X-ray images for pneumonia and other diseases. The project includes a backend API for predictions, a simple frontend, and tools for load testing and performance analysis.

---

## 📺 Video Demo

Watch the full demo on YouTube:  
[![Watch the video](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](YOUR_YOUTUBE_DEMO_LINK_HERE)

---

## 🌍 Live Demo

Try the live backend:  
[https://chest-xray-classifier-1.onrender.com](https://chest-xray-classifier-1.onrender.com)

---

## 📝 Project Description

This project provides an end-to-end pipeline for automated chest X-ray disease classification, featuring:
- A convolutional neural network (CNN) trained on chest X-ray images
- A REST API for inference using FastAPI/Flask
- Batch and single-image prediction endpoints
- Load testing using Locust
- Example code for data loading, training, and prediction

---

## 🚀 Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO
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
    uvicorn app:app --reload
    # or
    python app.py
    ```

5. **Test the API:**
    - Visit [http://localhost:8000/docs](http://localhost:8000/docs) to see the API documentation.
    - Use `/predict` to upload and classify images.

6. **Load Testing with Locust:**
    ```bash
    locust -f locustfile.py --host=http://localhost:8000
    # Open http://localhost:8089 to configure and start your test
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

- **Tool:** Locust
- **Target URL:** `https://chest-xray-classifier-1.onrender.com`
- **Config Example:**  
    - Users: 10  
    - Ramp up rate: 2 users/second

**Sample Output:**
```
Requests per second (RPS): 3.2
Average Response Time: 450 ms
Failure Rate: 0%
CPU Usage: 65%
```
*(Replace with your actual results if needed.)*

---

## 📎 Useful Links

- [YouTube Demo](YOUR_YOUTUBE_DEMO_LINK_HERE)
- [Live Demo](https://chest-xray-classifier-1.onrender.com)
- [API Docs](https://chest-xray-classifier-1.onrender.com/docs)

---

## 🛠️ Technologies Used

- Python, TensorFlow/Keras, FastAPI/Flask, Locust

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE)
