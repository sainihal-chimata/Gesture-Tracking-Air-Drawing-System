import cv2
import mediapipe as mp
import numpy as np
cam=cv2.VideoCapture(0)
hands=mp.solutions.hands
draw=mp.solutions.drawing_utils
hand_detector=hands.Hands()
status,result=cam.read()
height,width,_=result.shape
canvas=np.zeros((height,width,3),dtype=np.uint8)
prev_x=None
prev_y=None
mode="draw"
current_color=(0,255,0)
while True:
    status,result=cam.read()
    result=cv2.flip(result,1)
    colorchange=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
    results=hand_detector.process(colorchange)
    if results.multi_hand_landmarks:
        for handone in results.multi_hand_landmarks:
            draw.draw_landmarks(result,handone,hands.HAND_CONNECTIONS)
            index_finger=handone.landmark[8]
            ix=index_finger.x
            iy=index_finger.y
            iz=index_finger.z
            thumb_finger=handone.landmark[4]
            tx=thumb_finger.x
            ty=thumb_finger.y
            tz=thumb_finger.z
            distance=((ix-tx)**2+(iy-ty)**2)**0.5
            pixel_x=int(ix*width)
            pixel_y=int(iy*height)
            if prev_x is not None and prev_y is not None:
                movement=((pixel_x-prev_x)**2+(pixel_y-prev_y)**2)**0.5
                if distance<0.1 and movement>5:
                    if mode=="erase":
                        cv2.line(canvas,(prev_x,prev_y),(pixel_x,pixel_y),(0,0,0),20)
                    if mode=="draw":
                        cv2.line(canvas,(prev_x,prev_y),(pixel_x,pixel_y),current_color,20)
            prev_x=pixel_x
            prev_y=pixel_y
    else:
        prev_x=None
        prev_y=None
    final=cv2.add(result,canvas)
    cv2.imshow("frame",final)
    key=cv2.waitKey(1)
    key=key & 0xFF
    if key==ord('c'):
        canvas=np.zeros((height,width,3),dtype=np.uint8)
    if key==ord('q') or key==27:
        break
    if key==ord('e'):
        mode="erase"
    if key==ord('d'):
        mode="draw"
    if key==ord('1'):
        current_color=(0,255,0)
    if key==ord('2'):
        current_color=(255,0,0)
    if key==ord('3'):
        current_color=(0,0,255)
cam.release()
cv2.destroyAllWindows()
