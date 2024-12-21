
# FenderApronDetection

🚗 **Real-Time Fender Apron Defect Detection System**

This repository contains a system for detecting and classifying fender apron defects, such as cracks, rust, and good conditions. Designed for automotive applications, the system is built using YOLOv8 object detection and Streamlit for interactive visualization.
Live Demo available on : https://fender-apron-detection-systems.streamlit.app/

## Folder Structure

```plaintext
fender-apron-Defect-Detection-1
├── test                # Test dataset for evaluation
├── train               # Training dataset
├── valid               # Validation dataset
├── data.yaml           # Dataset configuration file for YOLO
├── runs
│   └── train
│       ├── fender_apron_model    # Trained YOLOv8 model version 1 (used for detection)
│       │   └── weights
│       │       └── best.pt       # Final weights file for YOLO model
│       └── fender_apron_model2   # Trained YOLOv8 model version 2
├── app.py              # Streamlit application for real-time detection
├── FenderApronDetection.ipynb    # Jupyter Notebook for model training
├── requirements.txt    # Python dependencies
├── yolo11n.pt          # YOLOv8 model (alternative version)
└── yolo8m.pt           # YOLOv8 medium model (default)
```

## Key Features

- **Real-Time Detection**: Perform real-time defect detection using your device’s camera.
- **Image Upload**: Upload images for automated defect analysis.
- **Defect Types**: Detects cracks, rust, and good fender aprons.
- **Interactive Metrics**: Displays detection counts, confidence levels, and processing times.
- **Optimized Models**: Utilizes YOLOv8 for high-accuracy detection and inference.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ibrahimhaykal/Fender-Apron-Detection.git
   cd FenderApronDetection/fender-apron-Defect-Detection-1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## How It Works

### Real-Time Detection (app.py)
- **Camera Mode**: Uses your webcam for real-time fender apron defect detection.
- **Image Upload Mode**: Upload an image, and the system analyzes defects with YOLOv8 models.
- **Insights**: Displays detected objects, total detection counts, and model confidence.

### Training (FenderApronDetection.ipynb)
- The training process utilizes YOLOv8 for training models based on labeled datasets in the `train` and `valid` directories.
- Model results and metrics are saved under the `runs/train` folder.

## Dataset
- The dataset is split into `train`, `valid`, and `test` folders.
- The `data.yaml` file contains configurations and class names for YOLOv8.

## Models
- The repository includes two pre-trained YOLO models:
  - `fender_apron_model/weights/best.pt`: Default model for detection.
  - `yolo8m.pt`: YOLOv8 medium-weight model.

## Results
- Model performance metrics (e.g., mAP, precision) are logged during the training process.

## Future Improvements
- Integrate additional defect categories.
- Enhance training with a larger dataset and more variative.

## Authors
- Ibrahim Haykal Alatas
- Fatur Rahman Zaki

**Built with passion for the Intelligent Systems course ✨**
