import socket
import pyaudio
import sys
import numpy as np
import copy
from mfcc_htk44100 import mfcc_htk
import setting

p = pyaudio.PyAudio()

# Client setting
Server_IP = socket.gethostbyname(socket.gethostname())
Sound_Port = 20000


# Pyaudio setting
CHUNK = 1024
WIDTH = 2
CHANNELS_MIC = 1
CHANNELS_SPEAKER = 2
RATE = 44100
bufferSize = 4096
Input = 0
Output = 0
mfcc_data = []
streamedAudio = np.zeros(8820, dtype=np.int16)
audio_set = np.array([0.0])
htk = mfcc_htk(44100)
mode = 3

soundAddressPort = (Server_IP, Sound_Port)

sound_sock = socket.socket(socket.AF_INET,  # Internet
                           socket.SOCK_DGRAM)  # UDP


sound_sock.bind(soundAddressPort)


def callback(in_data):
    global streamedAudio
    global mfcc_data
    global wav
    global RATE
    global htk

    sig = np.fromstring(in_data, 'Int16')

    sig = np.concatenate((streamedAudio, sig))

    if sig.shape[0] > int(RATE * 0.2 * 5):
        streamedAudio = sig[int(-1.0 * RATE * 0.2 * 2):]  # use only the last 0.2 sec of sound array
    else:
        streamedAudio = sig

    print('i')  # to check the ratio between audio and face gen, best is 1:1

    mfcc_feat = htk.run(sig[int(-1.0 * RATE * 0.22):])
    mfcc_feat = mfcc_feat[:20, 1:]
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)  # expand the dimension into (1,1,1,12,20)* or (1,1,1,12,20)* unsure
    mfcc_data = np.array(audio_set)

    return mfcc_data


# Stream Setting

pa = p.open(format=pyaudio.paInt16,
            channels=CHANNELS_MIC,
            rate=RATE,
            input=False,
            output=True,
            frames_per_buffer=CHUNK)

print("Recording")

# Streaming
while True:
    in_data, address = sound_sock.recvfrom(bufferSize)
    pa.write(in_data, CHUNK)
    mfcc_data = callback(in_data)

    # print(mfcc_data)
