import cv2
import face_recognition
import numpy as np  

img_elon = face_recognition.load_image_file('E:\Open CV\Face Recognition\TestImages\elon musk.jpg')
img_elon = cv2.cvtColor(img_elon,cv2.COLOR_BGR2RGB)
img_prabhas = face_recognition.load_image_file('E:\Open CV\Face Recognition\TestImages\prabhas.jpg')
img_prabhas = cv2.cvtColor(img_prabhas,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img_elon)[0]
encodeElon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocPrabhas = face_recognition.face_locations(img_prabhas)[0]
encodePrabhas = face_recognition.face_encodings(img_prabhas)[0]
cv2.rectangle(img_prabhas,(faceLocPrabhas[3],faceLocPrabhas[0]),(faceLocPrabhas[1],faceLocPrabhas[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodePrabhas)
faceDis = face_recognition.face_distance([encodeElon],encodePrabhas)
print(results,faceDis)
cv2.putText(img_prabhas,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Elon Musk",img_elon)
cv2.imshow("Prabhas", img_prabhas)
cv2.waitKey(0)