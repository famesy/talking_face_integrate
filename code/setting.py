import pyaudio
import sys
p = pyaudio.PyAudio()


def GetInputDeviceInfo():
    print("Input Device: ")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            n = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print("Input Device id ", i,"-", n.encode("utf8").decode("cp950", "ignore"))
    return "----------------------------------------------------------"


def GetOutputDeviceInfo():
    print("Output Device: ")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
            n = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print("Output Device id ", i,"-", n.encode("utf8").decode("cp950", "ignore"))
    return "----------------------------------------------------------"


def device_setting():
    if len(sys.argv) == 4:
        Input = int(sys.argv[2])
        Output = int(sys.argv[3])
        return [Input, Output]

    elif (len(sys.argv) == 3) and (int(sys.argv[2]) == 0):
        Input = None
        Output = None
        return [Input, Output]

    elif len(sys.argv) == 2:
        print(GetInputDeviceInfo())
        Input = int(input("Type 'Mic' Index : "))
        print(GetOutputDeviceInfo())
        Output = int(input("Type 'Sound Output' Index : "))
        return [Input, Output]

    else:
        raise IOError("Argument Error")
