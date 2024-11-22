import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.9, trackCon=0.1):
        self.mode = mode
        self.maxHands = maxHandsabout 150
        
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,self.detectionCon, self.trackCon, )
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDraw_hands = self.mpHands.Hands()

    def findHands(self,img, imgRGB, draw=False):
        result=self.mpDraw_hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handles in result.multi_hand_landmarks:
                if draw==True:
                    self.mpDraw.draw_landmarks(img, handles, self.mpHands.HAND_CONNECTIONS)
        return(img)
    def highlight2(self,img,imgRGB,lst):
        lmlst=[]
        result = self.mpDraw_hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handles in result.multi_hand_landmarks:
                for id, lm in enumerate(handles.landmark):
                    if id in lst:
                        h, w, c = imgRGB.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                        lmlst.append([str(id), str(cx), str(cy)])
        return (img,lmlst)
    def retlmlst_hands(self,imgRGB,lst):
        lmlst=[]
        result = self.mpDraw_hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handles in result.multi_hand_landmarks:
                for id, lm in enumerate(handles.landmark):
                    if id in lst:
                        h, w, c = imgRGB.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlst.append([str(id), str(cx), str(cy)])
        return (lmlst)


def main():
    ptime=0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1, complexity=1, detectionCon=0.9, trackCon=0.1)
    try:
        while True:
            ctime=time.time()
            fps=1/(ctime-ptime)
            ptime=ctime
            success, img = cap.read()
            imgRGV = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #detector.highlight2(img, imgRGV, [0,5])
            llist=detector.retlmlst_hands(imgRGV,[0,5])
            #llist=detector.retlmlst_hands(imgRGV,[4,8])
            if len(llist)!=0:
                print(llist)
            detector.findHands(img,imgRGV,True)
            cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)
            cv2.imshow('img', img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()