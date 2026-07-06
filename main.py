import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import av

st.title("Gesture Air Drawing System")

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

if 'hand_detector' not in st.session_state:
    st.session_state.hand_detector = hands.Hands()
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'prev_x' not in st.session_state:
    st.session_state.prev_x = None
if 'prev_y' not in st.session_state:
    st.session_state.prev_y = None
if 'mode' not in st.session_state:
    st.session_state.mode = "draw"
if 'current_color' not in st.session_state:
    st.session_state.current_color = (0, 255, 0)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Draw Mode (D)"):
        st.session_state.mode = "draw"
with col2:
    if st.button("Erase Mode (E)"):
        st.session_state.mode = "erase"
with col3:
    if st.button("Green (1)"):
        st.session_state.current_color = (0, 255, 0)
with col4:
    if st.button("Blue (2)"):
        st.session_state.current_color = (255, 0, 0)
with col5:
    if st.button("Red (3)"):
        st.session_state.current_color = (0, 0, 255)

if st.button("Clear Canvas (C)"):
    if st.session_state.canvas is not None:
        st.session_state.canvas = np.zeros_like(st.session_state.canvas)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    result = cv2.flip(img, 1)
    height, width, _ = result.shape

    if st.session_state.canvas is None or st.session_state.canvas.shape != result.shape:
        st.session_state.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    colorchange = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    results = st.session_state.hand_detector.process(colorchange)

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

            if st.session_state.prev_x is not None and st.session_state.prev_y is not None:
                movement = ((pixel_x - st.session_state.prev_x)**2 + (pixel_y - st.session_state.prev_y)**2)**0.5
                if distance < 0.1 and movement > 5:
                    if st.session_state.mode == "erase":
                        cv2.line(st.session_state.canvas, (st.session_state.prev_x, st.session_state.prev_y), (pixel_x, pixel_y), (0, 0, 0), 20)
                    if st.session_state.mode == "draw":
                        cv2.line(st.session_state.canvas, (st.session_state.prev_x, st.session_state.prev_y), (pixel_x, pixel_y), st.session_state.current_color, 20)
            
            st.session_state.prev_x = pixel_x
            st.session_state.prev_y = pixel_y
    else:
        st.session_state.prev_x = None
        st.session_state.prev_y = None

    final = cv2.add(result, st.session_state.canvas)
    return av.VideoFrame.from_ndarray(final, format="bgr24")

webrtc_streamer(
    key="air-drawing",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)