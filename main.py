import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import pyautogui as gui


########
wCam, hCam = 1280, 720
frameR = 150  # Frame reduction
smoothening = 20
########
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(detectionCon=0.8, maxHands=1)
wScr, hScr = gui.size()
x1, y1 = 0, 0

pTime = 0
plocX, plocY = 0, 0     # previous location
clocX, clocY = 0, 0     # current location
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Tip of index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8]
        x2, y2 = lmList[12]
    try:
        # Check fingers up
        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 0, 255), 2)
        # Moving Mode: Only index finger
        if fingers == [0, 1, 0, 0, 0]:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # smoothening
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # move cursor
            # print(wScr - x3, y3)
            gui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Scroll Mode: 1st and 2nd finger up
        '''if fingers == [0, 1, 1, 0, 0]:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # smoothening
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # move cursor
            # print(wScr - x3, y3)
            gui.scroll(plocY - clocY)
            # cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY'''

        # Left Click Mode: 1st, 2nd and 3rd finger up
        if fingers == [0, 1, 1, 1, 0]:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                gui.click()

        # Right Click Mode: 1st, 2nd, 3rd and 4th finger up
        if fingers == [0, 1, 1, 1, 1]:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                gui.click(button='right')

        # Zoom mode: thumb and 1st finger up
        if fingers == [1, 1, 0, 0, 0]:
            lengthZ, img, lineInfo = detector.findDistance(4, 8, img)
            new_length = np.interp(lengthZ, (25, 350), (0, 100))
            # smoothness
            new_length = smoothening * round(new_length/smoothening)
    except:
        pass
    # Flip image
    img = cv2.flip(img, 1)

    # Frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS:{int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
