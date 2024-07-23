import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images_list = []
class_names = []
my_list = os.listdir(path)
print(my_list)
for cl in my_list:
    current_image = cv2.imread(f"{path}/{cl}")
    images_list.append(current_image)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def mark_attendance(name):
    with open('attendance.csv', "r+") as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name}, {date_string}")

encode_list_known = find_encodings(images_list)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_current_frame = face_recognition.face_locations(imgS)
    encode_current_frame = face_recognition.face_encodings(imgS, face_current_frame)

    for encode_face, face_location in zip(encode_current_frame, face_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_distance = face_recognition.face_distance(encode_list_known, encode_face)
        print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_names[match_index].capitalize()
            print(name)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2 *4, x1 *4
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6, y2-6), cv2.FONT_ITALIC, 1, (255,255,255), 2)
            mark_attendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)

# face_location = face_recognition.face_locations(img_andria)[0]
# encode_andria = face_recognition.face_encodings(img_andria)[0]
# cv2.rectangle(img_andria,(face_location[3], face_location[0]), (face_location[1], face_location[2]), (255, 0, 255),2)
#
# face_location_test = face_recognition.face_locations(img_test)[0]
# encode_test = face_recognition.face_encodings(img_test)[0]
# cv2.rectangle(img_test,(face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255),2)
#
# results = face_recognition.compare_faces([encode_andria], encode_test)
# face_distance = face_recognition.face_distance([encode_andria], encode_test)