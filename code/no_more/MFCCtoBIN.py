from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy
import os
import torch
import wave
# directory where we your .wav files are


directoryName = '''C:/Users/Ryuusei/PycharmProjects/XPRIZE/DANV-master'''  # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = directoryName + "/testbin/binResult"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory
for filename in os.listdir(directoryName):
    wf = wave.open('0572_0019_0003.wav', 'rb')
    sig = wf.readframes(16000*4)
    sig = sig[1::2]
    sig = numpy.fromstring(sig, 'Int16')
    rate = 44100
    print(sig.shape)
    x = 1
    n = 10
    # get mfcc
    for i in range(0,sig.shape[0],int(sig.shape[0]/n)):
        mfcc_feat = zip(*mfcc(sig[i:i+int(rate * 0.2)],
                              rate,
                              winlen=0.025,
                              winstep=0.0095,
                              numcep=12,
                              nfilt=13,
                              nfft=2048,
                              lowfreq=300,
                              highfreq=3700,
                              preemph=0.97,
                              ceplifter=22,
                              appendEnergy=False))
        mfcc_feat = numpy.stack([numpy.array(i) for i in mfcc_feat])
        print(mfcc_feat.shape)

        cc = numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(mfcc_feat, axis=0), axis=0), axis=0)
        # print([cc])
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        if mfcc_feat.shape == (12,20):
            print('yes')
            x += 1
            cctBytes = bytearray(cc)
            newFile = open("test" + str(x) + ".bin", "wb")
            newFile.write(cctBytes)
