import wave, struct
import matplotlib.pyplot as plt
import numpy as np

def readwav(file):
    wav = wave.open(file)
    framerate = wav.getframerate()
    nchannels = wav.getnchannels()
    nframes = wav.getnframes()
    binary_data = wav.readframes(nframes)

    fmt = ''
    for _ in range(0,nframes):
        fmt = fmt + 'hh'

    # convert data van binary naar integers
    integer_data = struct.unpack(fmt, binary_data)

    return([framerate, nchannels, nframes, integer_data])

# Implement filters hier
def filter(data):
    res = data
    return res

def plot_data(data):
    framerate = data[0]
    nchannels = data[1]
    nframes = data[2]
    data = data[3]

    # Array voor de x as
    time = np.arange(0,nframes/framerate,1/framerate)

    # Neem alleen de data van kanaal 1
    data_left = []
    for i in range(len(data)):
        if i % 2 == 1:
            data_left.append(data[i])

    plt.plot(time, data_left)
    plt.xlabel('time[s]')
    plt.ylabel('Amplitude ')
    plt.show()

def main():
    data = filter(readwav('multiple birds.wav'))
    # print(data)
    plot_data(data)
    # plot_fourier(data)


if __name__== '__main__':
    main()