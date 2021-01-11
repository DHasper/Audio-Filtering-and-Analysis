import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
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
        fmt = fmt + 'h' * nchannels

    integer_data = struct.unpack(fmt, binary_data)
    if nchannels == 2:
        integer_data = integer_data[::2]

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

def plot_fourier(start, duration, *args):
    # Vul de assen
    axes = []
    for data in args:
        # Neem een subset van de data
        audio_data = data[3][start:start+duration]

        samplerate = data[0]
        intervals=len(audio_data)

        # Fourier transformatie
        sp = abs(2/intervals*np.fft.fft(audio_data))
        freq = np.fft.fftfreq(intervals,1/samplerate)

        # Alleen de eerste helft van het data is bruikbaar
        freq=freq[0:int(len(audio_data)/2)]
        sp=sp[0:int(len(audio_data)/2)]
        axes.append([freq, sp, data[4]])

    # The c variabele word gebruikt om een kleur te geven
    c = 0
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
    for axis in axes:
        plt.plot(axis[0], axis[1], colors[c], label=axis[2])
        c += 1
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude F')
    plt.show()

def plot_data(samplerate, nframes, data):
    # Array voor de x as
    time = np.arange(0,nframes/samplerate,1/samplerate)

    plt.plot(time, data)
    plt.xlabel('time[s]')
    plt.ylabel('Amplitude')
    plt.show()

def write_wav(samplerate, data):
    with wave.open('output.wav', "w") as f:
        f.setnchannels(1)
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
    b, a = signal.butter(2, [2400 / nyq, 3200 / nyq], btype='band')

    # Filter data in een nieuwe list
    data_filtered = data.copy()
    data_filtered[3] = iir_filter(data_filtered[3], a, b)

    # Plot fourier
    plot_fourier(64100, 10000, data, data_filtered)

    # Schrijf naar bestand
    write_wav(data_filtered[0], data_filtered[3])

if __name__== '__main__':
    main()