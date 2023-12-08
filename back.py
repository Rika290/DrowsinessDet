# FACE DETECTION
import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
v=cv2.VideoCapture('i.gif')
m=cv2.CascadeClassifier('face.xml')
mask=load_model('Mask.h5',compile=False)
while True:
    flag,frame=v.read()
    if flag:
        face=m.detectMultiScale(frame)
        for (x,y,l,w) in face:
            image=frame[y:y+w,x:x+l]
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image=(image/127.5)-1        
            pred=mask.predict(image)[0][0]
            if(pred>0.9):
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),5)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),5)
        cv2.namedWindow('W',cv2.WINDOW_NORMAL)
        cv2.imshow('W',frame)
        k=cv2.waitKey(1)
        if(k==ord('a')):
            break
    else:
        break
v.release()
cv2.destroyAllWindows()
