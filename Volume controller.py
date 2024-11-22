import cv2
import time
import math
import numpy as np
import HandTrackingModule as Htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


ptime = 0
cap = cv2.VideoCapture(0)

detector = Htm.HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume = cast(volume,POINTER(IAudioEndpointVolume))

#volume.GetMute()
#volume.GetMasterVolumeLevel()
#volumerange=volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
#min_volume=volumerange[0]
#max_volume=volumerange[1]


start_time=time.time()
end_time=start_time+6
try:
    while time.time()<=end_time:
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        success, img = cap.read()
        imgRGV = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        llst=detector.retlmlst_hands(imgRGV,[4,8])
        if len(llst)!=0:
            print(llst)
            x1,y1=eval(llst[0][1]),eval(llst[0][2])
            x2,y2=eval(llst[1][1]),eval(llst[1][2])
            cx,cy=(x1+x2)//2,(y1+y2)//2
            distance=math.sqrt((x1-x2)**2+(y1-y2)**2)
            cv2.circle(img,(x1,y1),5,(0,0,255),2)
            cv2.circle(img,(x2,y2),5,(0,0,255),2)
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.circle(img,(cx,cy),5,(0,0,255),2)
            print(distance)
            #min-20 max 197
            if distance<30:
                cv2.circle(img,(cx,cy),5,(0,255,0),2)
            volum=np.interp(distance,[20,150],[-45,0])
            print(volum)
            volume.SetMasterVolumeLevel(volum, None)
            vol=np.interp(distance,[20,150],[0,100])
            cv2.putText(img, 'Volume-' + str((round(volume.GetMasterVolumeLevelScalar() * 100))), (500, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.putText(img,'fps:'+str(int(fps)),(30,30),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
        cv2.putText(img, 'Volume-', (500, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255),2)

        cv2.imshow('img', img)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()