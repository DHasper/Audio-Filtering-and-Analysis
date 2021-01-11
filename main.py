import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from butterworth import butter_filter, butter
from datetime import datetime

def readwav(file):
    name = file
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

    return([framerate, nchannels, nframes, integer_data, name]) 

def iir_filter(x,a,b):
    Nsamples=len(x)
    y=[0]*Nsamples
    Na=len(a)
    Nb=len(b)
    for i in range(Nb,Nsamples):
        sumbx=0
        for j in range(0,Nb):
            if i-j >= 0:
                sumbx+=b[j]*x[i-j]
        sumay=0
        for k in range(1,Na):
            if i-k>=0:
                sumay+=a[k]*y[i-k]
        y[i]=(sumbx-sumay)/a[0]
    return(y)   

def plot_fourier(*args):
    start=64100
    duration=1000
    stop=start+duration
    axes = []

    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

    for data in args:
        # Neem een subset van de data
        samplerate = data[0]
        audio_data = data[3][start:stop]
        intervals=len(audio_data)

        # Fourier transformatie
        sp = abs(2/intervals*np.fft.fft(audio_data))
        freq = np.fft.fftfreq(intervals,1/samplerate)

        # Alleen de eerste helft van het data is bruikbaar
        freq=freq[0:int(len(audio_data)/2)]
        sp=sp[0:int(len(audio_data)/2)]
        axes.append([freq, sp, data[4]])

    c = 0
    for axis in axes:
        plt.plot(axis[0], axis[1], colors[c], label=axis[2])
        c += 1
    # plt.yscale('log')
    plt.legend()
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

def write_wav(samplerate, data, lowcut, highcut, name):
    output = name + " filtered.wav"
    with wave.open(output, "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        for frame in data:
            frame = struct.pack('<h', round(frame))
            f.writeframesraw(frame)

def main():
    # Lees wav file
    data = readwav('multiple birds.wav')

    # Bereken a, b coefficienten
    nyq = 0.5 * data[0]
    b, a = signal.butter(2, [1200 / nyq, 1600 / nyq], btype='band')

    # Filter data
    data_filtered = data.copy()
    data_filtered[3] = iir_filter(data_filtered[3], a, b)

    # Teken plot
    plot_fourier(data, data_filtered)
    
    # plot_data(nframes, samplerate, data)
    # data_filtered = butter_filter(audio_data, samplerate, 400, 3000, 2)
    # write_wav(samplerate, data_filtered, 1200, 1600, 'house sparrow')
    # step = 100
    # global lowcut, highcut
    # while step <= 2000:
    #     for i in range(1000, 4000):
    #         if i % 100 == 0:
    #             lowcut = i
    #             highcut = i + step
    #             if highcut <= 4000:
    #                 data_filtered = butter_filter(audio_data, samplerate, lowcut, highcut, 5)
    #                 write_wav(samplerate, data_filtered)
    #     step = step + 100

if __name__== '__main__':
    main()