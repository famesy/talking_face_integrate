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
import Train_Pic as tp
from mfcc_htk8000 import mfcc_htk
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

# stream variable
CHUNK = 400
WIDTH = 2
CHANNELS = 1
RATE = 8000
mfcc_data = []
p = pyaudio.PyAudio()

t = time.time()


def face_cap(): # mouth positioning -> cap -> save -> resize
    cap = cv2.VideoCapture(1)

    while True :
        ret,frame = cap.read()
        ret,frame2 = cap.read()
        cv2.rectangle(frame2, (270, 350), (370, 400), (0, 255, 0), 3)  # mouth position
        cv2.imshow('image', frame2)

        if cv2.waitKey(1) == ord("x"):
            cv2.imwrite('./demo_images/test_sample0.jpg',frame)  # face cap will save your face as test_sample0.jpg every time
            cv2.imwrite('./demo_images/cam/cam' + str(len(os.listdir('./demo_images/cam'))) + '.jpg', frame)  # for backing up face that can be use later
            print("Save")
            break
    cap.release()
    img = cv2.imread('./demo_images/test_sample0.jpg', cv2.IMREAD_UNCHANGED)

    Fromx = 150
    Fromy = 100
    h = 450
    w = 350
    crop_img = img[Fromy:Fromy+h, Fromx:Fromx+w]
    output = cv2.resize(crop_img, (256,256), interpolation= cv2.INTER_AREA)  # input pic for face_gen() and face_train()is 256*256
    cv2.imwrite('./demo_images/test_sample0.jpg',output)


def face_train(test_dataloader, n, do=True):  # train the mouth movement to match user
    if do is True:
        global mode
        global model

        print('Interference available')
        if mode != 0:  # test image

            for i2, data in enumerate(test_dataloader):
                if i2 < n:
                    A = cv2.imread('./0572_0019_0003/Video/'+str(i2 + 2)+'.jpg')
                    A = A/255
                    A = torch.from_numpy(A)
                    A.resize_(data['A'].size()).copy_(data['A'])
                    data['A'] = A
                    model.set_test_input(data)
                    model.test_train()
                else:
                    break

        elif mode == 0:  # your face from capfromVid.py

            n = 10
            for i2, data in enumerate(test_dataloader):
                if i2 < n:
                    A = cv2.imread('./train_face/Pic/'+str(i2)+'.jpg')
                    A = A/255
                    A = torch.from_numpy(A)
                    A.resize_(data['A'].size()).copy_(data['A'])
                    data['A'] = A
                    data['B_audio'] = torch.autograd.Variable(torch.from_numpy(np.array(data['B_audio'])).float())
                    model.set_test_input(data)
                    model.test_train()
                else:
                    break
    elif do is False:  # not gonna train. let it be
        pass


def face_gen(test_dataloader):  # face generating function
    enum = list(enumerate(test_dataloader))
    print("* recording")
    while True:
        k = 1
        for i in range(0, k):
            start = time.time()
            mfcc_data_torch = torch.autograd.Variable(torch.from_numpy(np.array(mfcc_data)).float())

            data = {}
            dic = enum[i][1]
            data['A'] = dic['A']
            data['A_path'] = dic['A_path']
            if 1:
                if mode == 2:
                    data['B_audio'] = dic['B_audio']
                else:
                    data['B_audio'] = mfcc_data_torch[0]

                model.set_test_input(data)
                model.test()
                visuals = model.get_current_visuals()
                im_rgb = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
                output = cv2.resize(im_rgb, (512, 512), interpolation=cv2.INTER_AREA)  # enlarge generated picture into 512*512
                cv2.imshow('TEST', output)
                cv2.waitKey(1)
            end = time.time()
            print('GENERATE', end - start, 'sec')  # compare ratio of generating with 'i' from callback() , best is 1:1


def open_stream():  # open local streaming for getting input sound
    stream = p.open(format=pyaudio.paInt16,  # p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,  # get sound
                    output=True,  # sound playback True=sound check, False=prevent echo
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)  # callback function will be use


def callback(in_data, frame_count=None, time_info=None, status=None):  # callback to collect sound array every 0.2 sec and convert it into MFCC
    global streamedAudio
    global mfcc_data
    global wav
    global RATE
    global mode
    global htk

    if mode == 2:
        noi = 1000.0 * np.random.normal(0, 0.1, CHUNK)
        noi = noi.astype(np.int16)
        sig = copy.deepcopy(wav[:CHUNK])
        wav = np.concatenate((wav[CHUNK:], wav[:CHUNK]))

    else:
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
    audio_set = np.expand_dims(np.expand_dims(np.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)  # expand the dimension into (1,1,1,12,20)* or (1,1,1,12,20)* unsure
    mfcc_data = np.array(audio_set)
    return (out_data, pyaudio.paContinue)


# start here
print('Mode : ')
mode = int(input())  # 0=face_cap+stream, 1=image+local stream, 2=image+load audio, 3=image+UDP stream
if mode == 0:
    test_nums = [0]
else:
    print('Input Pic : ')
    test_nums = [int(input())]
print('Train Count : ')
num = int(input())

if mode == 0:
    tp.face_train_pic()  # 1st train your face
    # face_cap()  # already have data but want now face
    open_stream()

    A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
    test_folder = Test_VideoFolder(root='./train_face/Bin', A_path=A_path, config=opt)
    test_dataloader = DataLoader(test_folder, batch_size=1)

    face_train(test_dataloader, num, do=True)
    face_gen(test_dataloader)

elif mode == 1:
    open_stream()

    A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
    test_folder = Test_VideoFolder(root='./0572_0019_0003/Audio', A_path=A_path, config=opt)
    test_dataloader = DataLoader(test_folder, batch_size=1)

    face_train(test_dataloader, num, do=True)
    face_gen(test_dataloader)

elif mode == 2:  # not really work, audio will get really fast and sounds jib jib like birds,then face will stay still nothing happen
    audio_file = '0572_0019_0003.wav'
    wf = wave.open(audio_file, 'rb')
    wav = wf.readframes(RATE*500)
    wav = wav[1::2]
    wav = np.fromstring(wav, 'Int16')

    open_stream()

    A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
    test_folder = Test_VideoFolder(root='./0572_0019_0003/Audio', A_path=A_path, config=opt)
    test_dataloader = DataLoader(test_folder, batch_size=1)

    face_train(test_dataloader, num, do=True)
    face_gen(test_dataloader)


elif mode == 3:  # still in development, super kak now, callback doesn't work so I call callback() and then BOOM. Its sounds like your jack is loose and your voice is delayed by 2 sec
    Server_IP = socket.gethostbyname(socket.gethostname())  # get your IP
    Server_Port = 20000
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP

    sock.bind((Server_IP, Server_Port))

    # Stream Setting
    bufferSize = 4096
    paSPEAKER = p.open(format=pyaudio.paInt16,
                       channels=CHANNELS,
                       rate=RATE,
                       input=False,
                       output=True,
                       frames_per_buffer=CHUNK)

    A_path = os.path.join(opt.test_A_path, 'test_sample' + str(test_nums[0]) + '.jpg')
    test_folder = Test_VideoFolder(root='./0572_0019_0003/Audio', A_path=A_path, config=opt)
    test_dataloader = DataLoader(test_folder, batch_size=1)

    face_train(test_dataloader, num, do=True)

    enum = list(enumerate(test_dataloader))
    print("* recording")
    while True:

        # mfcc_data, address = sock.recvfrom(bufferSize)  # buffer size is 4096 bytes
        # mfcc_data = np.frombuffer(mfcc_data, 'Int16')

        data_IN, address = sock.recvfrom(bufferSize)
        paSPEAKER.write(data_IN, CHUNK)
        callback(data_IN)
        k = 1

        for i in range(0, k):
            start = time.time()
            mfcc_data_torch = torch.autograd.Variable(torch.from_numpy(np.array(mfcc_data)).float())

            data = {}
            dic = enum[i][1]
            data['A'] = dic['A']
            data['A_path'] = dic['A_path']
            if 1:
                if mode == 2:
                    data['B_audio'] = dic['B_audio']
                else:
                    data['B_audio'] = mfcc_data_torch[0]

                model.set_test_input(data)
                model.test()
                visuals = model.get_current_visuals()
                im_rgb = cv2.cvtColor(visuals['fake_audio_B_0'], cv2.COLOR_RGB2BGR)
                output = cv2.resize(im_rgb, (512, 512), interpolation=cv2.INTER_AREA)  # enlarge generated picture into 512*512
                cv2.imshow('TEST', output)
                cv2.waitKey(1)
            end = time.time()
            print('GENERATE', end - start, 'sec')  # compare ratio of generating with 'i' from callback() , best is 1:1



