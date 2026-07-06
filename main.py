import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import av

st.title("Gesture Air Drawing System")

# Initialize MediaPipe objects once globally
hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

# Use a persistent cache to hold states across threads safely
@st.cache_resource
def get_hand_detector():
    return hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

hand_detector = get_hand_detector()

# Thread-safe global state containers
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'mode' not in st.session_state:
    st.session_state.mode = "draw"
if 'current_color' not in st.session_state:
    st.session_state.current_color = (0, 255, 0)

# Local trackpad variables for the background callback thread
prev_x = None
prev_y = None

# Sidebar control panel (UI elements work fine with session_state)
st.sidebar.title("Controls")
mode = st.sidebar.radio("Select Mode", ("Draw", "Erase"), index=0)
color = st.sidebar.selectbox("Select Color", ("Green", "Blue", "Red"), index=0)

if st.sidebar.button("Clear Canvas"):
    if st.session_state.canvas is not None:
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)

# Map UI selections to operational variables
current_color = (0, 255, 0) if color == "Green" else (255, 0, 0) if color == "Blue" else (0, 0, 255)
operational_mode = mode.lower()

def video_frame_callback(frame):
    global prev_x, prev_y
    
    img = frame.to_ndarray(format="bgr24")
    result = cv2.flip(img, 1)
    height, width, _ = result.shape

    if st.session_state.canvas is None or st.session_state.canvas.shape != result.shape:
        st.session_state.canvas = np.zeros((height, width, 3), dtype=np.uint8)

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
                    if operational_mode == "erase":
                        cv2.line(st.session_state.canvas, (prev_x, prev_y), (pixel_x, pixel_y), (0, 0, 0), 20)
                    elif operational_mode == "draw":
                        cv2.line(st.session_state.canvas, (prev_x, prev_y), (pixel_x, pixel_y), current_color, 20)
            
            prev_x = pixel_x
            prev_y = pixel_y
    else:
        prev_x = None
        prev_y = None

    final = cv2.add(result, st.session_state.canvas)
    return av.VideoFrame.from_ndarray(final, format="bgr24")

webrtc_streamer(
    key="air-drawing",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceTransportPolicy": "all"
    }
)