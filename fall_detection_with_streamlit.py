import cv2
import requests
import streamlit as st
import tempfile
import numpy as np
import os

# Configuration
API_KEY = "df7e8IGIzCSNyzpl0wPh"
PROJECT_ID = "fall-detection-mbldh"
MODEL_VERSION = "1"
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

def infer_frame(frame):
    """Send frame to Roboflow API for inference"""
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    response = requests.post(
        INFERENCE_URL,
        files={"file": img_encoded.tobytes()},
        data={"name": "video_frame"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.text}")
        return None

def annotate_frame(frame, predictions, conf_threshold):
    """Annotate frame with bounding boxes and labels"""
    annotated_frame = frame.copy()
    
    if not predictions or "predictions" not in predictions:
        return annotated_frame
    
    for pred in predictions["predictions"]:
        confidence = pred["confidence"]
        
        # Skip low confidence detections
        if confidence < conf_threshold:
            continue
            
        # Extract bounding box coordinates and info
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        class_name = pred["class"].lower()
        
        # Flip the labels (as per your original logic)
        if class_name == "fall":
            display_label = "stand"
            color = (0, 255, 0)  # Green for stand
        elif class_name == "stand":
            display_label = "fall"
            color = (0, 0, 255)  # Red for fall
        else:
            display_label = class_name
            color = (255, 255, 0)  # Yellow for unknown
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, 
                     (x - w//2, y - h//2), 
                     (x + w//2, y + h//2), 
                     color, 5)
        
        # Draw label
        label = f"{display_label} ({confidence:.2f})"
        cv2.putText(annotated_frame, label, 
                   (x - w//2, y - h//2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5)
    
    return annotated_frame

def process_uploaded_video(uploaded_file, conf_threshold):
    """Process uploaded video file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name
    
    # Open video
    cap = cv2.VideoCapture(temp_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    output_path = tempfile.mktemp(suffix='_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_placeholder = st.empty()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Get predictions and annotate frame
            predictions = infer_frame(frame)
            annotated_frame = annotate_frame(frame, predictions, conf_threshold)
            
            # Write to output video
            out.write(annotated_frame)
            
            # Show current frame (every 10th frame to avoid too much updating)
            if frame_count % 10 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                    channels="RGB",
                    caption=f"Frame {frame_count}"
                )
        
        # Cleanup
        cap.release()
        out.release()
        
        # Read the processed video for download
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes, frame_count, output_path
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, 0, None
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def process_webcam_image(camera_input, conf_threshold):
    """Process image from webcam"""
    # Convert camera input to opencv format
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Get predictions and annotate
    predictions = infer_frame(frame)
    annotated_frame = annotate_frame(frame, predictions, conf_threshold)
    
    return annotated_frame

def main():
    """Main Streamlit application"""
    st.title("Fall Detection System with Roboflow")
    st.write("Upload a video or use webcam to detect falls using AI")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Input source selection
    st.subheader("📹 Choose Input Source")
    option = st.radio(
        "Select input method:", 
        ["Upload Video", "Webcam"],
        help="Choose between uploading a video file or using your webcam"
    )
    
    if option == "Upload Video":
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"**File:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            # Process button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Processing video... This may take a while."):
                    video_bytes, total_frames, output_path = process_uploaded_video(
                        uploaded_file, conf_threshold
                    )
                    
                    if video_bytes:
                        st.success(f"Processing complete! Analyzed {total_frames} frames.")
                        
                        # Display result video
                        st.subheader("📹 Annotated Result")
                        st.video(video_bytes)
                        
                        # Download button
                        st.download_button(
                            label="Download Annotated Video",
                            data=video_bytes,
                            file_name=f"fall_detection_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                        
                        # Clean up output file
                        if output_path and os.path.exists(output_path):
                            os.unlink(output_path)
    
    elif option == "Webcam":
        st.subheader("Webcam Analysis")
        st.write("Take a photo to analyze for fall detection")
        
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            with st.spinner("Analyzing image..."):
                annotated_frame = process_webcam_image(camera_input, conf_threshold)
                
                if annotated_frame is not None:
                    st.subheader("Analysis Result")
                    st.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        caption="Annotated image with fall detection"
                    )
    
    # Information section
    with st.expander("How it works"):
        st.write("""
        **Detection Legend:**
        - 🟢 **Green boxes**: Person standing
        - 🔴 **Red boxes**: Person falling  
        - 🟡 **Yellow boxes**: Unknown class
        - **Numbers in parentheses**: Confidence scores (0.0 - 1.0)
        
        **Instructions:**
        1. Choose your input method (Upload Video or Webcam)
        2. Adjust the confidence threshold if needed
        3. Upload a video file or take a photo
        4. Click 'Analyze' and wait for processing
        5. View results and download if needed
        
        **Note:** The model labels are flipped in this implementation:
        - When model detects "fall" → displayed as "stand"
        - When model detects "stand" → displayed as "fall"
        """)

if __name__ == "__main__":
    main()