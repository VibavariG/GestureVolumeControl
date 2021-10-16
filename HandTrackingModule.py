import cv2
import mediapipe as mp
import time  # to check the frame rate

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode    #obj will have its own variable
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # have to do b4 we start using this model
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)  # creating an obj 'hands', using default params
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # send rgb image to hands obj, coz that obj takes only rgb imgs
        self.results = self.hands.process(imgRBG)  # process frame and give hand landmarks and handedness
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # method provided by mediapipe
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # draw on og img coz we're displaying og img
        return img




    def findPosition(self, img, handNo=0, draw=True):
        lmList = []    #landmark list
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)  # prints every point coordinates in x,y - these are in decimals, but we need pixels
                h, w, c = img.shape  # height, width, channels
                cx, cy = int(lm.x * w), int(lm.y * h)  # finding pos in pixels
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                  cv2.circle(img, (cx, cy), 15, (255,0, 255), cv2.FILLED)
        return lmList


def main():
    # frameRate, fps
    pTime = 0  # previous time
    cTime = 0  # current time
    # create video object
    cap = cv2.VideoCapture(0)  # "http://56.79.179.161:8080/video"
    detector = handDetector()
    while True:
        success, img = cap.read()  # will give us our frames
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0 :
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

#if we are running this script
if __name__ == "__main__":
    main()         #whatever we write in main() will be a dummy code to show what this module can do