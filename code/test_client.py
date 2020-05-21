import time
from Options_all import BaseOptions
from torch.utils.data import DataLoader
import os
import cv2
import torch
import util.util as util
import numpy as np
import pyaudio
import wave
import copy
import socket
import struct
import pickle
import Train_Pic as tp
from mfcc_htk44100 import mfcc_htk
import Test_Gen_Models.Test_Audio_Model as Gen_Model
from Dataloader.Test_load_audio import Test_VideoFolder

opt = BaseOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1

# global variable
streamedAudio = np.zeros(8820, dtype=np.int16)
audio_set = np.array([0.0])
htk = mfcc_htk(44100)
model = Gen_Model.GenModel(opt)
model, _, start_epoch = util.load_test_checkpoint('./checkpoints/101_DAVS_checkpoint.pth.tar', model)

# Pyaudio setting
p = pyaudio.PyAudio()
CHUNK = 2048
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


def callback(in_data, frame_count=None, time_info=None,
             status=None):  # callback to collect sound array every 0.2 sec and convert it into MFCC
    global streamedAudio
    global mfcc_data
    global wav
    global RATE
    global htk
    global bufferSize
    global address
    global CHUNK

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

    return out_data, pyaudio.paContinue


def face_train(test_dataloader, n):  # train the mouth movement to match user

    global mode
    global model
    if n > 0:
        print('Interference available')
        if mode != 0:  # test image
            for i2, data in enumerate(test_dataloader):
                if i2 < n:
                    A = cv2.imread('./0572_0019_0003/Video/' + str(i2 + 2) + '.jpg')
                    A = A / 255
                    A = torch.from_numpy(A)
                    A.resize_(data['A'].size()).copy_(data['A'])
                    data['A'] = A
                    model.set_test_input(data)
                    model.test_train()
                else:
                    break

        elif mode == 0:  # your face from capfromVid.py

            for i2, data in enumerate(test_dataloader):
                if i2 < n:
                    A = cv2.imread('./train_face/Pic/' + str(i2) + '.jpg')
                    A = A / 255
                    A = torch.from_numpy(A)
                    A.resize_(data['A'].size()).copy_(data['A'])
                    data['A'] = A
                    data['B_audio'] = torch.autograd.Variable(torch.from_numpy(np.array(data['B_audio'])).float())
                    model.set_test_input(data)
                    model.test_train()
                else:
                    break


def face_gen(test_dataloader):  # face generating function
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    enum = list(enumerate(test_dataloader))
    print("* recording")

    while True:
        audio = sound.read(CHUNK)
        sound_sock.sendto(audio, soundAddressPort)
        k = 1
        for i in range(0, k):
            start = time.time()
            mfcc_data_torch = torch.autograd.Variable(torch.from_numpy(np.array(mfcc_data)).float())

            data = {}
            dic = enum[i][1]
            data['A'] = dic['A']
            data['A_path'] = dic['A_path']
            if 1:
                data['B_audio'] = mfcc_data_torch[0]
                model.set_test_input(data)
                model.test()
                visuals = model.get_current_visuals()
                output = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
                # output = cv2.resize(output, (128, 128), interpolation=cv2.INTER_AREA)

                result, output = cv2.imencode('.jpg', output, encode_param)
                data = pickle.dumps(output, 0)
                size = len(data)
                Pic_sock.sendall(struct.pack(">L", size) + data)
            end = time.time()
            print('GENERATE', end - start, 'sec')  # compare ratio of generating with 'i' from callback() , best is 1:1


# Client setting
Server_IP = '25.91.156.227'
Sound_Port = 20000
Pic_Port = 8485

soundAddressPort = (Server_IP, Sound_Port)
PicAddrressPort = (Server_IP, Pic_Port)

sound_sock = socket.socket(socket.AF_INET,  # Internet
                           socket.SOCK_DGRAM)  # UDP
Pic_sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_STREAM)  # UDP

# start here
print('Input Pic : ')
test_nums = [int(input())]
print('Train Count : ')
num = int(input())

# Stream Setting

pa = p.open(format=pyaudio.paInt16,
            channels=CHANNELS_MIC,
            rate=RATE,
            input=True,
            output=False,
            frames_per_buffer=CHUNK,
            stream_callback=callback)

sound = p.open(format=p.get_format_from_width(WIDTH),
               channels=CHANNELS_MIC,
               rate=RATE,
               input=True,
               output=False,
               frames_per_buffer=CHUNK)

A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
test_folder = Test_VideoFolder(root='./0572_0019_0003/Audio', A_path=A_path, config=opt)
test_dataloader = DataLoader(test_folder, batch_size=1)

enum = list(enumerate(test_dataloader))

# Streaming
face_train(test_dataloader, num)

Pic_sock.connect(PicAddrressPort)
connection2 = Pic_sock.makefile('wb')

face_gen(test_dataloader)
