import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Set the title of the Streamlit app
st.title("License Image and Video Processing")

# Allow users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load YOLO model
try:
    model = YOLO('best.pt')  # Replace with the path to your trained YOLO model
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")


def predict_and_save_image(path_test_car, output_image_path):
    try:
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def predict_and_plot_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            st.error("Failed to initialize video writer.")
            return None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None


if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input:
        temp_input.write(uploaded_file.getbuffer())
        input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_output:
        output_path = temp_output.name

    st.write("Processing...")
    result_path = process_media(input_path, output_path)

    if result_path:
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            with open(result_path, 'rb') as video_file:
                st.video(video_file.read())
        else:
            st.image(result_path)
