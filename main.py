import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.title("Gesture Air Drawing System")

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

@st.cache_resource
def get_hand_detector():
    return hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

hand_detector = get_hand_detector()

st.sidebar.title("Configuration")
ui_mode = st.sidebar.radio("Select Mode", ("Draw", "Erase"), index=0)
ui_color = st.sidebar.selectbox("Select Color", ("Green", "Blue", "Red"), index=0)

mode = ui_mode.lower()
current_color = (0, 255, 0) if ui_color == "Green" else (255, 0, 0) if ui_color == "Blue" else (0, 0, 255)

uploaded_file = st.file_uploader("Upload a video clip of your hand gestures to process:", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cam = cv2.VideoCapture(tfile.name)
    
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    prev_x, prev_y = None, None
    
    st.write("Processing video frames...")
    frame_placeholder = st.empty()
    
    while cam.isOpened():
        status, result = cam.read()
        if not status:
            break
            
        result = cv2.flip(result, 1)
        colorchange = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(colorchange)
        
        if results.multi_hand_landmarks:
            for handone in results.multi_hand_landmarks:
                draw.draw_landmarks(result, handone, hands.HAND_CONNECTIONS)
                
                index_finger = handone.landmark[8]
                ix, iy = index_finger.x, index_finger.y
                
                thumb_finger = handone.landmark[4]
                tx, ty = thumb_finger.x, thumb_finger.y
                
                distance = ((ix - tx)**2 + (iy - ty)**2)**0.5
                pixel_x = int(ix * width)
                pixel_y = int(iy * height)

                if prev_x is not None and prev_y is not None:
                    movement = ((pixel_x - prev_x)**2 + (pixel_y - prev_y)**2)**0.5
                    if distance < 0.1 and movement > 5:
                        if mode == "erase":
                            cv2.line(canvas, (prev_x, prev_y), (pixel_x, pixel_y), (0, 0, 0), 20)
                        elif mode == "draw":
                            cv2.line(canvas, (prev_x, prev_y), (pixel_x, pixel_y), current_color, 20)
                
                prev_x = pixel_x
                prev_y = pixel_y
        else:
            prev_x = None
            prev_y = None
            
        final = cv2.add(result, canvas)
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(final_rgb, channels="RGB", use_container_width=True)
        
    cam.release()
    st.success("Processing complete!")