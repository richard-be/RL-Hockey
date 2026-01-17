import numpy as np

def generate_colored_noise(N, beta):
    spectrum = np.zeros(N, dtype=np.complex128)
    k = np.arange(1, N//2+1)
    f = k/N
    re = np.random.normal(size=len(k))
    im = np.random.normal(size=len(k))
    scale = f**(-beta /2)
        
    spectrum[k] = (re + 1j * im) * scale
    spectrum[-k] = np.conjugate(spectrum[k])
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=N)
    return normalize(signal)

def normalize(signal):
    signal = signal - np.mean(signal)
    signal = signal/np.sqrt(np.var(signal))
    return signal