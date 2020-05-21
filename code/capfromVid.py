import cv2
import msvcrt
import numpy as np
import pyaudio
import copy
from mfcc_htk44100 import mfcc_htk


streamedAudio = np.zeros(8820, dtype=np.int16)
audio_set = np.array([0.0])
htk = mfcc_htk(44100)


def callback(in_data, frame_count, time_info, status):
    global streamedAudio
    global mfcc_data
    global wav
    global RATE
    global mode
    global htk

    sig = np.fromstring(in_data, 'Int16')
    out_data = copy.deepcopy(sig)
    sig = np.concatenate((streamedAudio, sig))

    if sig.shape[0] > int(RATE * 0.2 * 5):
        streamedAudio = sig[int(-1.0 * RATE * 0.2 * 2):]
    else:
        streamedAudio = sig
    print('i')
    mfcc_feat = htk.run(sig[int(-1.0 * RATE * 0.22):])
    mfcc_feat = mfcc_feat[:20, 1:]
    # mfcc_feat = mfcc_matlab
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)
    mfcc_data = np.array(audio_set)
    return (out_data, pyaudio.paContinue)


cap = cv2.VideoCapture(0)

CHUNK = 4000
WIDTH = 2
CHANNELS = 1
RATE = 44100
p = pyaudio.PyAudio()
mfcc_data = []
stream = p.open(format=pyaudio.paInt16,  # p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

n = 100  # amount of face cap
sound_data = []
while True :
    ret,frame = cap.read()
    ret,frame2 = cap.read()
    cv2.rectangle(frame2, (270, 350), (370, 400), (0, 255, 0), 3)
    cv2.imshow('image', frame2)
    cv2.waitKey(1)

    if msvcrt.kbhit():
        for i in range(n):
            ret,frame = cap.read()
            ret,frame2 = cap.read()
            cv2.rectangle(frame2, (270, 350), (370, 400), (0, 255, 0), 3)
            mfcc_data = np.array(mfcc_data)
            sound_data.append(mfcc_data[0])
            cv2.imshow('image',frame2)
            cv2.waitKey(1)
            cv2.imwrite('./train_face/Pic/' + str(i) + '.jpg',frame)
            print("Save")
        break

stream.close()
cap.release()

for i in range(n):
    img = cv2.imread('./train_face/Pic/' + str(i) + '.jpg', cv2.IMREAD_UNCHANGED)
    Fromx = 150
    Fromy = 50
    h = 400
    w = 350
    crop_img = img[Fromy:Fromy+h, Fromx:Fromx+w]
    output = cv2.resize(crop_img, (256,256), interpolation= cv2.INTER_AREA)
    cv2.imwrite('./train_face/Pic/' + str(i) + '.jpg',output)

    newFileBytes = sound_data[i]
    # print(newFileBytes.size())
    newFileByteArray = bytearray(newFileBytes)
    newFile = open("./train_face/Bin/" + str(i) + ".bin", "wb")
    newFile.write(newFileByteArray)
    newFile.close()


cv2.destroyAllWindows()



