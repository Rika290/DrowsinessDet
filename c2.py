import cv2
import time
import numpy as np
import streamlit as st
from keras.models import load_model
model = load_model("keras_model_1.h5", compile=False) # eyes
model_a=load_model("keras_model.h5", compile=False)  # mouth

st.set_page_config(page_title='Drowsiness Detection',page_icon= 'https://i.pinimg.com/originals/0a/1f/f1/0a1ff1e3e1ffd7750aec75d572429b00.jpg')
st.title('WELCOME TO DROWSINESS DETECTION 😴⚠🔊')
choice=st.sidebar.selectbox('Dashboard',('Home','Camera','IP Camera','Upload file or link'))
st.sidebar.image('https://clipart-library.com/newhp/139-1395363_49-truck-driver-male-asian-clipart-female-truck.png')
if(choice=='Home'):
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt-f_zzNu1uEFoZP8ejwjABpsRrQ4QvcnL_w&usqp=CAU')
elif(choice=='Camera'):
    pic = st.camera_input("Take a picture")

#if picture:
    #a=st.number_input('Enter 0 for accessing your Camera') # since, 0 is for camera, input is taken in number format..thats why, st.number_input and not ss text_input.
    window=st.empty()
    #b=st.button('Start')
    if pic:
        v=cv2.VideoCapture(pic) # as a is 0 here, int(a) is taken.. 
        f_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        e_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
        m_cas=cv2.CascadeClassifier('mouth.xml')
        b1=st.button('Stop')
        if b1:
            v.release()
            st.rerun()

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        while True:
            flag,frame=v.read()
            if flag:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                face=f_cas.detectMultiScale(frame,1.3,5)

                for(x,y,w,h) in face:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    roi_a = gray[y:y+h, x:x+w]
                    roi_b = frame[y:y+h, x:x+w]
            
                    eyes=e_cas.detectMultiScale(roi_a)                                      
                    for (ex,ey,ew,eh) in eyes:
                        image = cv2.resize(roi_b, (224, 224), interpolation=cv2.INTER_AREA)
                        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                        image = (image / 127.5) - 1
                        # Predicts the model
                        prediction_a = model.predict(image)
                        index = np.argmax(prediction_a)                
                        confidence_score_a = prediction_a[0][index]
                
                        print("Confidence Score:", str(np.round(confidence_score_a * 100))[:-2], "%")
                        print('Pred Score:',confidence_score_a)
                        cv2.putText(frame,'Pred Score(eyes):'+str(confidence_score_a),(10,20), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_a)<=0.8:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                            cv2.putText(frame,"EYES:- close",(10,40), font, 1,(255,0,0),1,cv2.LINE_AA)
                            html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards                        

                            
                        else:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                            cv2.putText(frame,"EYES:- open",(10,40), font, 1,(0,0,255),1,cv2.LINE_AA)
                    mouth=m_cas.detectMultiScale(frame,1.7,11)
                    for(x,y,w,h) in mouth:
                        y = int(y - 0.15*h)
                        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                        img = (img / 127.5) - 1
                        # Predicts the model
                        prediction_b = model_a.predict(img)
                        index = np.argmax(prediction_b)                
                        confidence_score_b = prediction_b[0][index]

                        print("Confidence Score (mouth):", str(np.round(confidence_score_b * 100))[:-2], "%")
                        print('Pred Score(mouth):',confidence_score_b)
                        cv2.putText(frame,'Pred Score(mouth):'+str(confidence_score_b),(10,70), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_b)<=0.8:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                
                            cv2.putText(frame,"MOUTH:- open",(10,90), font, 1,(0,0,255),1,cv2.LINE_AA)
                            html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards
                            

                        else:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                            cv2.putText(frame,"MOUTH:- close",(10,90), font, 1,(0,255,0),1,cv2.LINE_AA)                                                            

                window.image(frame,channels='BGR')    
elif(choice=='IP Camera'):    
    a=st.text_input('Enter the WebCam link') # since, 0 is for camera, input is taken in number format..thats why, st.number_input and not ss text_input.
    window=st.empty()
    b=st.button('Upload a video')
    if b:
        v=cv2.VideoCapture(a) # as a is 0 here, int(a) is taken.. 
        f_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        e_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
        m_cas=cv2.CascadeClassifier('mouth.xml')
        b1=st.button('Stop')
        if b1:
            v.release()
            st.rerun()

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        while True:
            flag,frame=v.read()
            if flag:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                face=f_cas.detectMultiScale(frame,1.3,5)

                for(x,y,w,h) in face:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    roi_a = gray[y:y+h, x:x+w]
                    roi_b = frame[y:y+h, x:x+w]
            
                    eyes=e_cas.detectMultiScale(roi_a)                                      
                    for (ex,ey,ew,eh) in eyes:
                        image = cv2.resize(roi_b, (224, 224), interpolation=cv2.INTER_AREA)
                        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                        image = (image / 127.5) - 1
                        # Predicts the model
                        prediction_a = model.predict(image)
                        index = np.argmax(prediction_a)                
                        confidence_score_a = prediction_a[0][index]
                
                        print("Confidence Score:", str(np.round(confidence_score_a * 100))[:-2], "%")
                        print('Pred Score:',confidence_score_a)
                        cv2.putText(frame,'Pred Score(eyes):'+str(confidence_score_a),(10,20), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_a)<=0.8:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                            cv2.putText(frame,"EYES:- close",(10,40), font, 1,(255,0,0),1,cv2.LINE_AA)
                            html_string = """

                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """
                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards

                            

                        else:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                            cv2.putText(frame,"EYES:- open",(10,40), font, 1,(0,0,255),1,cv2.LINE_AA)
                                
                    mouth=m_cas.detectMultiScale(frame,1.7,11)
                    for(x,y,w,h) in mouth:
                        y = int(y - 0.15*h)
                        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                        img = (img / 127.5) - 1
                        # Predicts the model
                        prediction_b = model_a.predict(img)
                        index = np.argmax(prediction_b)                
                        confidence_score_b = prediction_b[0][index]

                        print("Confidence Score (mouth):", str(np.round(confidence_score_b * 100))[:-2], "%")
                        print('Pred Score(mouth):',confidence_score_b)
                        cv2.putText(frame,'Pred Score(mouth):'+str(confidence_score_b),(10,70), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_b)<=0.8:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                
                            cv2.putText(frame,"MOUTH:- open",(10,90), font, 1,(0,0,255),1,cv2.LINE_AA)
                            html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """
                            #<source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards

                            

                        else:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                            cv2.putText(frame,"MOUTH:- close",(10,90), font, 1,(0,255,0),1,cv2.LINE_AA)                                                            

                window.image(frame,channels='BGR')
else:    
    a=st.text_input('Enter link or path of the video file') # since, 0 is for camera, input is taken in number format..thats why, st.number_input and not ss text_input.
    window=st.empty()
    b=st.button('Upload a video')
    if b:
        v=cv2.VideoCapture(a) # as a is 0 here, int(a) is taken.. 
        f_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        e_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
        m_cas=cv2.CascadeClassifier('mouth.xml')
        b1=st.button('Stop')
        if b1:
            v.release()
            st.rerun()

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        while True:
            flag,frame=v.read()
            if flag:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                face=f_cas.detectMultiScale(frame,1.3,5)

                for(x,y,w,h) in face:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    roi_a = gray[y:y+h, x:x+w]
                    roi_b = frame[y:y+h, x:x+w]
            
                    eyes=e_cas.detectMultiScale(roi_a)                                      
                    for (ex,ey,ew,eh) in eyes:
                        image = cv2.resize(roi_b, (224, 224), interpolation=cv2.INTER_AREA)
                        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                        image = (image / 127.5) - 1
                        # Predicts the model
                        prediction_a = model.predict(image)
                        index = np.argmax(prediction_a)                
                        confidence_score_a = prediction_a[0][index]
                
                        print("Confidence Score:", str(np.round(confidence_score_a * 100))[:-2], "%")
                        print('Pred Score:',confidence_score_a)
                        cv2.putText(frame,'Pred Score(eyes):'+str(confidence_score_a),(10,20), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_a)<=0.8:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                            cv2.putText(frame,"EYES:- close",(10,40), font, 1,(255,0,0),1,cv2.LINE_AA)
                            html_string = """

                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """
                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards

                            

                        else:
                            cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                            cv2.putText(frame,"EYES:- open",(10,40), font, 1,(0,0,255),1,cv2.LINE_AA)
                                
                    mouth=m_cas.detectMultiScale(frame,1.7,11)
                    for(x,y,w,h) in mouth:
                        y = int(y - 0.15*h)
                        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                        img = (img / 127.5) - 1
                        # Predicts the model
                        prediction_b = model_a.predict(img)
                        index = np.argmax(prediction_b)                
                        confidence_score_b = prediction_b[0][index]

                        print("Confidence Score (mouth):", str(np.round(confidence_score_b * 100))[:-2], "%")
                        print('Pred Score(mouth):',confidence_score_b)
                        cv2.putText(frame,'Pred Score(mouth):'+str(confidence_score_b),(10,70), font, 1,(255,255,255),1,cv2.LINE_AA)

                        if(confidence_score_b)<=0.8:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                
                            cv2.putText(frame,"MOUTH:- open",(10,90), font, 1,(0,0,255),1,cv2.LINE_AA)
                            html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """
                            #<source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards

                            

                        else:
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                            cv2.putText(frame,"MOUTH:- close",(10,90), font, 1,(0,255,0),1,cv2.LINE_AA)                                                            

                window.image(frame,channels='BGR')                        
                                                        
                
                                                        
