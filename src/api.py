from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
from prediction import load_trained_model, predict_image, predict_batch


app = FastAPI(title="Chest X-Ray Pneumonia API")
start_time = time.time()

# Load model at startup
MODEL_PATH = "models/best_model.h5"
CLASS_LABELS = {0: 'Normal', 1: 'Pneumonia'}
model = load_trained_model(MODEL_PATH)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def get_status():
    uptime = time.time() - start_time
    return {"status": "ok", "uptime_seconds": int(uptime)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
    try:
        label, confidence = predict_image(model, temp_path, CLASS_LABELS)
    except Exception as e:
        os.remove(temp_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    os.remove(temp_path)
    return {"predicted_label": label, "confidence": confidence}

@app.post("/predict_batch")
async def predict_batch_api(files: list[UploadFile] = File(...)):
    temp_paths = []
    results = []
    try:
        for file in files:
            contents = await file.read()
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(contents)
            temp_paths.append(temp_path)

        batch_results = predict_batch(model, temp_paths, CLASS_LABELS)
        results = batch_results
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
    return {"results": results}

@app.post("/upload")
async def upload_data(file: UploadFile = File(...), label: str = Form(...)):
    train_dir = f"data/new_uploads/{label}"
    os.makedirs(train_dir, exist_ok=True)
    file_path = os.path.join(train_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"File {file.filename} uploaded to {train_dir}"}

@app.post("/retrain")
def retrain_model_api():
    import subprocess
    result = subprocess.run(["python", "src/retrain.py"], capture_output=True, text=True)
    global model
    model = load_trained_model(MODEL_PATH)  # reload updated model
    return {
        "message": "Retraining completed.",
        "stdout": result.stdout,
        "stderr": result.stderr
    }




if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)