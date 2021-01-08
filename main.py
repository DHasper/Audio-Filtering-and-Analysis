import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from butterworth import butter_filter
from datetime import datetime

file = 'single bird'
highcut = 5000
lowcut = 0

def readwav(file):
    wav = wave.open(file)
    framerate = wav.getframerate()
    nchannels = wav.getnchannels()
    nframes = wav.getnframes()
    binary_data = wav.readframes(nframes)

    fmt = ''
    for _ in range(0,nframes):
        fmt = fmt + 'hh'

    # Convert data van binary naar integers
    integer_data = struct.unpack(fmt, binary_data)

    return([framerate, nchannels, nframes, integer_data]) 

def plot_fourier(nframes, samplerate, data):
    # Neem een subset van de data
    start=64100
    duration=1000
    stop=start+duration
    # data = data[start:stop]
    intervals=len(data)

    # Filter data
    Ts = 1/samplerate
    w_co=2*np.pi*1
    a=np.array([1+w_co*Ts,-1])
    b=np.array([w_co*Ts])
    print(a, b)

    data_filtered = butter_filter(data, samplerate, 1000, 4000, 5)

    # Fourier transformatie
    sp = abs(2/intervals*np.fft.fft(data))
    sp_filtered = abs(2/intervals*np.fft.fft(data_filtered))
    freq = np.fft.fftfreq(intervals,1/samplerate)

    # Alleen de eerste helft van het data is bruikbaar
    freq=freq[0:int(len(data)/2)]
    sp=sp[0:int(len(data)/2)]
    sp_filtered=sp_filtered[0:int(len(data_filtered)/2)]

    plt.plot(freq, sp,'r', freq, sp_filtered, 'b')
    # plt.yscale('log')
    plt.title('FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude F ')
    plt.show()

def plot_data(nframes, samplerate, data):
    # Array voor de x as
    time = np.arange(0,nframes/samplerate,1/samplerate)

    # Neem alleen de data van kanaal 1
    data_left = []
    for i in range(len(data)):
        if i % 2 == 1:
            data_left.append(data[i])

    plt.plot(time, data_left)
    plt.xlabel('time[s]')
    plt.ylabel('Amplitude ')
    plt.show()

def write_wav(samplerate, data):
    with wave.open("output/%s-%s-%s.wav" % (file, lowcut, highcut), "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        for frame in data:
            frame = struct.pack('<h', round(frame))
            f.writeframesraw(frame)

def main():
    data = readwav(file + '.wav')

    samplerate = data[0]
    nchannels = data[1]
    nframes = data[2]
    data = data[3]

    # plot_data(nframes, samplerate, data)
    # plot_fourier(nframes, samplerate, data)
    data_filtered = butter_filter(data, samplerate, lowcut, highcut, 5)
    write_wav(samplerate, data_filtered)
    # step = 200
    # global lowcut, highcut
    # for i in range(1000, 6000):
    #     if i % step == 0:
    #         lowcut = i
    #         highcut = i + step
    #         data_filtered = butter_filter(data, samplerate, lowcut, highcut, 5)
    #         write_wav(samplerate, data_filtered)

if __name__== '__main__':
    main()