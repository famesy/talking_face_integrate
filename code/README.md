# **Real Time Face Generation**

- [Reference paper](https://arxiv.org/abs/1807.07860)
- [Reference git](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS)
- [Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

### Requirement
- python 3.6
- pyAudio
- numpy 1.17.4
- tensorflow-gpu 1.11.0
- opencv2
- torch 1.2.0+cu92
- torchvision 0.4.0+cu92

*Now we still use a checkpoint from [Reference paper](https://arxiv.org/abs/1807.07860)*

*Use realtime_facegen44100.py if your record device have 44100 Hz and use 8000 if its 8000 Hz*

### Mode
##### Mode 0
- train model with your face first then use that model to generate face by local streaming voice

##### Mode 1
- use example image to generate face by local streaming voice

##### Mode 2
- use example image to generate face by .wav audio clip 
  - **still unusable : audio clip going too fast still generate face but no face movement**

##### Mode 3
- use example image to generate face by UDP streaming
  - **still unusable : voice delayed by 2 sec**
