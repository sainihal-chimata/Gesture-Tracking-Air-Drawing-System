import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Gesture Air Drawing System")

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

@st.cache_resource
def get_hand_detector():
    return hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

hand_detector = get_hand_detector()

class AppState:
    canvas = None
    prev_x = None
    prev_y = None
    mode = "draw"
    current_color = (0, 255, 0)

st.sidebar.title("Controls")
ui_mode = st.sidebar.radio("Select Mode", ("Draw", "Erase"), index=0)
ui_color = st.sidebar.selectbox("Select Color", ("Green", "Blue", "Red"), index=0)

AppState.mode = ui_mode.lower()
AppState.current_color = (0, 255, 0) if ui_color == "Green" else (255, 0, 0) if ui_color == "Blue" else (0, 0, 255)

if st.sidebar.button("Clear Canvas"):
    AppState.canvas = None
    AppState.prev_x = None
    AppState.prev_y = None

# Streamlit's native browser camera proxy
img_file_buffer = st.camera_input("Position your hand clearly in front of your camera to draw")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    result = cv2.flip(cv2_img, 1)
    height, width, _ = result.shape

    if AppState.canvas is None or AppState.canvas.shape != result.shape:
        AppState.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    colorchange = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(colorchange)

    if results.multi_hand_landmarks:
        for handone in results.multi_hand_landmarks:
            draw.draw_landmarks(result, handone, hands.HAND_CONNECTIONS)
            index_finger = handone.landmark[8]
            ix, iy = index_finger.x, index_finger.y
            thumb_finger = handone.landmark[4]
            tx, ty = thumb_finger.x, thumb_finger.y
            
            distance = ((ix-tx)**2+(iy-ty)**2)**0.5
            pixel_x = int(ix*width)
            pixel_y = int(iy*height)
            
            if AppState.prev_x is not None and AppState.prev_y is not None:
                movement = ((pixel_x-AppState.prev_x)**2+(pixel_y-AppState.prev_y)**2)**0.5
                if distance < 0.1 and movement > 5:
                    if AppState.mode == "erase":
                        cv2.line(AppState.canvas, (AppState.prev_x, AppState.prev_y), (pixel_x, pixel_y), (0, 0, 0), 20)
                    if AppState.mode == "draw":
                        cv2.line(AppState.canvas, (AppState.prev_x, AppState.prev_y), (pixel_x, pixel_y), AppState.current_color, 20)
            
            AppState.prev_x = pixel_x
            AppState.prev_y = pixel_y
    else:
        AppState.prev_x = None
        AppState.prev_y = None

    final = cv2.add(result, AppState.canvas)
    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), use_container_width=True)