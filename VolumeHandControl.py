import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
# library for increasing and decreasing volume -> pycaw by Andre Miras under MIT License
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########################
wCam, hCam = 640, 480
##########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)    # increasing the detection confidence to avoid strays

# pycaw usage code from github
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()           # -65: 0, -20: 26, -10: 51, 0: 100


minVolume = volumeRange[0]
maxVolume = volumeRange[1]
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)   # finding the hand & drawing the lines
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])                   # printing pixel pos of all landmarks in a loop
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2               # center of the line b/w thumb and index

        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)                     #blue

        # find distance b/w 2 points
        length = math.hypot(x2-x1, y2-y1)
        # print(length)
        # Hand Range 40 to 250
        # Volume Range -65 to 0
        # Convert Hand Range to Volume Range - use a numpy function

        vol = np.interp(length, [25, 230], [minVolume, maxVolume])
        volBar = np.interp(length, [25, 230], [400, 150])
        volPer = np.interp(length, [25, 230], [0, 100])
        trial = np.interp(volPer, [0, 100], [minVolume, maxVolume])
        # print(vol)
        volume.SetMasterVolumeLevel(trial, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)            #yellow

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (50, 425), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # print frame rate on img
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2. imshow("Image", img)
    cv2.waitKey(1)
