import socket
import pyaudio
import sys
import numpy as np
import copy
from mfcc_htk44100 import mfcc_htk
import setting

p = pyaudio.PyAudio()

# Client setting
Server_IP = '192.168.1.102'
Server_Port = 20000

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


def callback(in_data, frame_count, time_info, status):  # callback to collect sound array every 0.2 sec and convert it into MFCC
    global streamedAudio
    global mfcc_data
    # global wav
    global RATE
    global mode
    global htk

    # if mode == 2:
    #     noi = 1000.0 * np.random.normal(0, 0.1, CHUNK)
    #     noi = noi.astype(np.int16)
    #     sig = copy.deepcopy(wav[:CHUNK])
    #     wav = np.concatenate((wav[CHUNK:], wav[:CHUNK]))

    # else:
    #     sig = np.fromstring(in_data, 'Int16')
    sig = np.fromstring(in_data, 'Int16')
    out_data = copy.deepcopy(sig)
    sig = np.concatenate((streamedAudio, sig))

    if sig.shape[0] > int(RATE * 0.2 * 5):
        streamedAudio = sig[int(-1.0 * RATE * 0.2 * 2):]  # use only the last 0.2 sec of sound array
    else:
        streamedAudio = sig

    print('i')  # to check the ratio between audio and face gen, best is 1:1

    mfcc_feat = htk.run(sig[int(-1.0 * RATE * 0.22):])
    mfcc_feat = mfcc_feat[:20, 1:]
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0),
                               axis=0)  # expand the dimension into (1,1,1,12,20)* or (1,1,1,12,20)* unsure
    mfcc_data = np.array(audio_set)
    return (out_data, pyaudio.paContinue)


# Device setting
# device = setting.device_setting()
# Input = device[0]
# Output = device[1]

sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP

# Stream Setting

paMIC = p.open(format=p.get_format_from_width(WIDTH),
               channels=CHANNELS_MIC,
               rate=RATE,
               input=True,
               output=True,
               frames_per_buffer=CHUNK)

paMFCC = p.open(format=pyaudio.paInt16,
                channels=CHANNELS_MIC,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

serverAddressPort = (Server_IP, Server_Port)

print("Connect")

# Streaming
while True:
    try:
        # Send sound to server
        data_OUT = paMIC.read(CHUNK)  # Reading Data
        # mfcc_data = bytearray(mfcc_data)
        # sock.sendto(mfcc_data, serverAddressPort)  # Sending Data
        sock.sendto(data_OUT, serverAddressPort)
        # Receive sound from server
        # data_IN, address = sock.recvfrom(bufferSize)  # Recieve Data
        # paSPEAKER.write(data_IN, CHUNK)  # streaming Data

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break

    except Exception:
        print("Exception", sys.exc_info())
        break
print("Disconnect")
