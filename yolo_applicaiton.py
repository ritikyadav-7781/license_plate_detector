import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from moviepy.editor import VideoFileClip


# Set the title of the Streamlit app
st.title("YOLO Image and Video Processing")

# Allow users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load YOLO model
try:
    model = YOLO('best.pt')  # Replace with the path to your trained YOLO model
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

def predict_and_save_image(path_test_car, output_image_path):
    """
    Predicts and saves the bounding boxes on the given test image using the trained YOLO model.
    
    Parameters:
    path_test_car (str): Path to the test image file.
    output_image_path (str): Path to save the output image file.

    Returns:
    str: The path to the saved output image file.
    """
    try:
        results = model.predict(path_test_car, device='cpu')
        image = Image.open(path_test_car).convert("RGB")
        draw = ImageDraw.Draw(image)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 10), f'{confidence*100:.2f}%', fill="blue")
        image.save(output_image_path)
        return output_image_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_and_plot_video(video_path, output_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.

    Parameters:
    video_path (str): Path to the test video file.
    output_path (str): Path to save the output video file.

    Returns:
    str: The path to the saved output video file.
    """
    try:
        def process_frame(frame):
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            results = model.predict(np.array(image), device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                    draw.text((x1, y1 - 10), f'{conf*100:.2f}%', fill="red")
            return np.array(image)

        clip = VideoFileClip(video_path).fl_image(process_frame)
        clip.write_videofile(output_path, codec='libx264')
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def process_media(input_path, output_path):
    """
    Processes the uploaded media file (image or video) and returns the path to the saved output file.

    Parameters:
    input_path (str): Path to the input media file.
    output_path (str): Path to save the output media file.

    Returns:
    str: The path to the saved output media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing...")
        result_path = process_media(input_path, output_path)
        if result_path:
            if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_file = open(result_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.image(result_path)
    except Exception as e:
        st.error(f"Error uploading or processing file: {e}")
