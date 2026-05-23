# emergency_vechicle


## Project Overview

This project builds a multi-lane ambulance detection system using a convolutional neural network (CNN) and a Streamlit web interface. The app analyzes uploaded lane video files and detects whether an ambulance is present in each lane.

## Key Files

- `app.py` - Streamlit application for uploading videos and running ambulance detection using the trained model.
- `backend.py` - Training script for the CNN model; loads images from a dataset, trains the model, and saves it as `ambulance_cnn.keras`.
- `ambulance_cnn.keras` - Trained TensorFlow Keras model used by `app.py`.
- `carsdataset/` - Dataset directory structure used for training in `backend.py`.

## Features

- Upload multiple lane videos at once
- Detect ambulance presence per lane
- Display results clearly for each video lane

## Requirements

- Python 3.8+ recommended
- `tensorflow`
- `streamlit`
- `opencv-python`
- `numpy`

Install dependencies with:

```bash
pip install tensorflow streamlit opencv-python numpy
```

## Run the App

1. Make sure `ambulance_cnn.keras` is in the project root.
2. Run:

```bash
streamlit run app.py
```

3. Upload lane video files and review the detected ambulance results.

## Train the Model

If you want to retrain the model:

1. Update the dataset path in `backend.py` if needed.
2. Run:

```bash
python backend.py
```

3. The trained model will be saved as `ambulance_cnn.keras`.

## Notes

- `backend.py` currently points to a local dataset path in your archive. Change that path to match your dataset location if needed.
- The app uses a simple threshold (`pred > 0.5`) for ambulance detection.
- For best results, provide clear lane videos with visible ambulance imagery.
