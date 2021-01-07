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

    # Convert data van binary naar integers
    integer_data = struct.unpack(fmt, binary_data)

    return([framerate, nchannels, nframes, integer_data])

def plot_fourier(nframes, samplerate, data):

    # Neem een subset van de data
    start=10000
    duration=10000
    stop=start+duration
    data = data[start:stop]
    intervals=len(data)

    # Fourier transformatie
    sp = abs(2/intervals*np.fft.fft(data))
    freq = np.fft.fftfreq(intervals,1/samplerate)

    # Alleen de eerste helft van het data is bruikbaar
    freq=freq[0:int(len(data)/2)]
    sp=sp[0:int(len(data)/2)]

    plt.plot(freq, sp,'^r')
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

def main():
    data = readwav('multiple birds.wav')

    samplerate = data[0]
    nchannels = data[1]
    nframes = data[2]
    data = data[3]

    # plot_data(nframes, samplerate, data)
    plot_fourier(nframes, samplerate, data)


if __name__== '__main__':
    main()