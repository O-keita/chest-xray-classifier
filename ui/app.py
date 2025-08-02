import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np

API_URL = "https://chest-xray-classifier-1.onrender.com/"

st.title("Chest X-Ray Pneumonia Detection")

# Sidebar: Show API status
with st.sidebar:
    st.header("Model/API Status")
    try:
        status = requests.get(f"{API_URL}/status").json()
        st.write(f"Uptime: {status['uptime_seconds']} seconds")
        st.write(f"Status: {status['status']}")
    except Exception:
        st.error("Could not connect to backend.")

# 1. Predict single image
st.header("Predict Pneumonia from X-Ray")
uploaded_file = st.file_uploader("Upload an X-Ray Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())}
        )
        result = response.json()
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Prediction: {result['predicted_label']}")
            st.write(f"Confidence: {result['confidence']:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# 2. Batch prediction
st.header("Batch Prediction")
batch_files = st.file_uploader(
    "Upload Multiple X-Ray Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch"
)
if batch_files:
    files = [("files", (f.name, f.getvalue())) for f in batch_files]
    try:
        response = requests.post(f"{API_URL}/predict_batch", files=files)
        results = response.json()["results"]
        for res in results:
            if "error" in res:
                st.error(f"{res.get('image', 'Unknown')}: {res['error']}")
            else:
                st.write(f"{res.get('image', 'Unknown')}: {res['label']} (Confidence: {res['confidence']:.2f})")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# 3. Upload for retraining
st.header("Upload New Image for Retraining")
retrain_file = st.file_uploader("Upload Image for Retraining", type=["jpg", "jpeg", "png"], key="retrain")
label = st.selectbox("Label", ["Normal", "Pneumonia"])
if retrain_file and label:
    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (retrain_file.name, retrain_file.getvalue())},
            data={"label": label},
        )
        st.success(response.json()["message"])
    except Exception as e:
        st.error(f"Upload failed: {e}")

# 4. Trigger retraining
st.header("Trigger Model Retraining")
if st.button("Retrain Model"):
    try:
        response = requests.post(f"{API_URL}/retrain")
        st.info(response.json().get("message", "Retraining triggered."))
    except Exception as e:
        st.error(f"Could not start retraining: {e}")

# 5. Model Training History / Evaluation
st.header("Model Training History & Evaluation")

try:
    with open("models/history.json", "r") as f:
        history = json.load(f)

    epochs = np.arange(1, len(history["accuracy"]) + 1)
    best_epoch = 5  # Adjust based on your best epoch (zero-based index, so 5 means epoch 6 for users)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy
    axs[0, 0].plot(epochs, history["accuracy"], label="Train Accuracy")
    axs[0, 0].plot(epochs, history["val_accuracy"], label="Val Accuracy")
    axs[0, 0].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
    axs[0, 0].set_title("Accuracy")
    axs[0, 0].legend()

    # Loss
    axs[0, 1].plot(epochs, history["loss"], label="Train Loss")
    axs[0, 1].plot(epochs, history["val_loss"], label="Val Loss")
    axs[0, 1].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
    axs[0, 1].set_title("Loss")
    axs[0, 1].legend()

    # AUC
    axs[1, 0].plot(epochs, history.get("auc", []), label="Train AUC")
    axs[1, 0].plot(epochs, history.get("val_auc", []), label="Val AUC")
    axs[1, 0].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
    axs[1, 0].set_title("AUC")
    axs[1, 0].legend()

    # Precision
    axs[1, 1].plot(epochs, history.get("precision", []), label="Train Precision")
    axs[1, 1].plot(epochs, history.get("val_precision", []), label="Val Precision")
    axs[1, 1].axvline(best_epoch+1, color="g", linestyle="--", label="Best Epoch")
    axs[1, 1].set_title("Precision")
    axs[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Optionally, summarize best epoch metrics
    st.markdown(f"**Best Epoch (Early Stopping): {best_epoch+1}**")
    st.markdown(f"- Validation Accuracy: {history['val_accuracy'][best_epoch]:.3f}")
    st.markdown(f"- Validation Loss: {history['val_loss'][best_epoch]:.3f}")
    st.markdown(f"- Validation AUC: {history['val_auc'][best_epoch]:.3f}")
    st.markdown(f"- Validation Precision: {history['val_precision'][best_epoch]:.3f}")




except Exception as e:
    st.warning("Training history not found or error reading history.json.")
    st.text(f"Error: {e}")

