import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
from ultralytics import YOLO
import math
import cvzone
from PIL import Image
import time
import asyncio

# Page configuration
st.set_page_config(
    page_title="Fender Apron Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .uploadedFile {
        border: 2px dashed #ccc;
        padding: 2rem;
        border-radius: 10px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 1rem;
        text-align: center;
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("runs/train/fender_apron_model/weights/best.pt")

model = load_model()
classNames = ['Crack', 'Good', 'Rust']

def resize_image(image, target_size=(640, 640)):
    """Resize image maintaining aspect ratio and pad if necessary"""
    original_size = image.size
    ratio = float(target_size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])

    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    offset = (
        (target_size[0] - new_size[0]) // 2,
        (target_size[1] - new_size[1]) // 2
    )
    new_image.paste(image, offset)

    return new_image, offset

# Title
st.title("🔍 Fender Apron Detection System")
st.markdown("### Detect cracks, rust, and condition of fender aprons in real-time")

# Create tabs
tab1, tab2 = st.tabs(["📷 Camera Detection", "🖼️ Image Upload"])

# Camera tab
with tab1:
    st.markdown("### Real-time Detection")

    class VideoProcessor:
        def __init__(self):
            self.model = model
            self.classNames = classNames

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            height, width = img.shape[:2]
            target_height, target_width = 720, 1280

            pil_img = Image.fromarray(img)
            resized_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            img_array = np.array(resized_img)

            results = self.model(img_array, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(img_array, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(
                        img_array, f'{self.classNames[cls]} {conf}',
                        (max(0, x1), max(35, y1)), scale=1, thickness=1
                    )

            return av.VideoFrame.from_ndarray(img_array, format="bgr24")

    # WebRTC Configuration with Multiple STUN Servers
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.sipnet.ru:3478"]}
        ]}
    )

    # Camera selection with improved resolution handling
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280, "max": 1920},
                "height": {"min": 480, "ideal": 720, "max": 1080},
                "aspectRatio": {"ideal": 1.7777}  # 16:9 ratio
            },
            "audio": False
        },
        async_processing=True,
        sendback_audio=False,
    )

# Image upload tab
with tab2:
    st.markdown("### Upload Image")
    st.info("Images will be resized to 640x640 pixels for optimal detection")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.write(f"Original image size: {image.size}")

        resized_image, padding_offset = resize_image(image)
        img_array = np.array(resized_image)
        st.write(f"Processed image size: {resized_image.size}")

        start_time = time.time()
        results = model(img_array, stream=True)

        num_detections, max_conf = 0, 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                num_detections += 1
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img_array, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                max_conf = max(max_conf, conf)

                cls = int(box.cls[0])
                cvzone.putTextRect(
                    img_array, f'{classNames[cls]} {conf:.2f}',
                    (max(0, x1), max(35, y1)), scale=1, thickness=1
                )

        processing_time = (time.time() - start_time) * 1000

        st.image(img_array, caption='Processed Image (640x640)', use_column_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", f"{num_detections}")
        with col2:
            st.metric("Processing Time", f"{processing_time:.1f} ms")
        with col3:
            st.metric("Max Confidence", f"{max_conf * 100:.1f}%")

# Footer with system information and creator
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### System Information")
    st.info(
        """
        - This system uses YOLO object detection for fender apron analysis
        - All images are processed at 640x640 resolution for optimal detection
        - Camera access is handled through your browser's WebRTC implementation
        - You can select different cameras if multiple are available
        """
    )

with col2:  
    st.markdown("### About")  
    st.info(  
        """  
        Created by Ibrahim Haykal & Fatur Rahman Zaki 

        A real-time fender apron defect detection system for the automotive industry.  
        Developed for the Intelligent Systems course, it utilizes the YOLOv8 object detection framework and deep learning.  
        Trained on a dataset processed and augmented in Roboflow.  
        Dataset source: [GitHub](https://github.com/kapil-verma/Machine_part_defect-detection).  
        Efficient, accurate, and optimized for industrial applications.  
        """  
    )  

st.markdown(
    """
    <div class="footer">
        Created with 🔥 by Ibrahim Haykal & Fatur Rahman Zaki  © 2024
    </div>
    """,
    unsafe_allow_html=True
)
