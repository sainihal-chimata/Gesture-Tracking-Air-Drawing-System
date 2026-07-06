import gradio as gr
import cv2
import mediapipe as mp
import numpy as np

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hand_detector = hands.Hands()

# Global state persistency for the video stream loop
canvas = None
prev_x = None
prev_y = None

def process_frame(frame, mode, ui_color):
    global canvas, prev_x, prev_y
    
    if frame is None:
        return None
        
    # Gradio provides frames in RGB format
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = cv2.flip(result, 1)
    height, width, _ = result.shape
    
    if canvas is None or canvas.shape != result.shape:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
    # Map color options
    if ui_color == "Green":
        current_color = (0, 255, 0)
    elif ui_color == "Blue":
        current_color = (255, 0, 0)
    else:
        current_color = (0, 0, 255)
        
    operational_mode = mode.lower()
    colorchange = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(colorchange)
    
    if results.multi_hand_landmarks:
        for handone in results.multi_hand_landmarks:
            draw.draw_landmarks(result, handone, hands.HAND_CONNECTIONS)
            
            index_finger = handone.landmark[8]
            ix, iy = index_finger.x, index_finger.y
            
            thumb_finger = handone.landmark[4]
            tx, ty = thumb_finger.x, thumb_finger.y
            
            distance = ((ix-tx)**2 + (iy-ty)**2)**0.5
            pixel_x = int(ix * width)
            pixel_y = int(iy * height)
            
            if prev_x is not None and prev_y is not None:
                movement = ((pixel_x - prev_x)**2 + (pixel_y - prev_y)**2)**0.5
                if distance < 0.1 and movement > 5:
                    if operational_mode == "erase":
                        cv2.line(canvas, (prev_x, prev_y), (pixel_x, pixel_y), (0, 0, 0), 20)
                    if operational_mode == "draw":
                        cv2.line(canvas, (prev_x, prev_y), (pixel_x, pixel_y), current_color, 20)
            
            prev_x = pixel_x
            prev_y = pixel_y
    else:
        prev_x = None
        prev_y = None
        
    final = cv2.add(result, canvas)
    return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

def clear_canvas():
    global canvas, prev_x, prev_y
    canvas = None
    prev_x = None
    prev_y = None
    return None

# Build the real-time layout interface
with gr.Blocks() as demo:
    gr.Markdown("# Gesture Air Drawing System")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode_input = gr.Radio(["Draw", "Erase"], label="Select Mode", value="Draw")
            color_input = gr.Dropdown(["Green", "Blue", "Red"], label="Select Color", value="Green")
            clear_btn = gr.Button("Clear Canvas")
            
        with gr.Column(scale=2):
            # This handles the continuous live webcam frame-by-frame data piping natively
            webcam_stream = gr.Image(sources=["webcam"], streaming=True, label="Live Feed")

    # Hook the streaming feed directly to your logic function
    webcam_stream.stream(
        fn=process_frame, 
        inputs=[webcam_stream, mode_input, color_input], 
        outputs=webcam_stream,
        time_per_frame=0.03 # Target ~30 frames per second smoothly
    )
    
    clear_btn.click(fn=clear_canvas, inputs=[], outputs=[])

demo.launch()