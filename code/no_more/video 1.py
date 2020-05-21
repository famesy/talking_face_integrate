import cv2
import os

PATH = 'C:/Users/Ryuusei/Videos/Captures/'

for File in os.listdir(PATH):
    print(File)
    if File.startswith('OK'):
        cap = cv2.VideoCapture(PATH+File)
        if cap.isOpened() is False:
            print("Error opening video stream or file")
            break
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                cv2.imshow('Example',frame)
                cv2.waitKey(1)
            elif ret is False:
                break

