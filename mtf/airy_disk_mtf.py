
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
#from scipy.fft import fft, fftfreq, rfft, dct
import numpy.fft as fft
import airy_mtf_improve

def calculate_airy_disk_profile(wavelength=550e-9, f_stop=4.12, r_max=1e-3, num_points=10000):
    # Calculate Numerical Aperture (NA)
    NA = 1 / (2 * f_stop)
    
    # Calculate wavenumber k
    k = (2 * np.pi / wavelength) * NA

    # Define radial distances (in meters) from 0 to r_max
    r_values = np.linspace(0, r_max, num_points)

    # Calculate intensity distribution (normalized to peak intensity)
    intensity = (2 * j1(k * r_values) / (k * r_values))**2
    intensity[0] = 1  # Define the value at r = 0 (avoid division by zero)
    
    return r_values, intensity

def calculate_1D_MTF(r_values, intensity):
    # Calculate the Fourier Transform of the 1D Airy disk profile to get the OTF
    OTF_1D = fft.fft(intensity)

    # Calculate the 1D MTF as the magnitude of the OTF
    MTF_1D = np.abs(OTF_1D)

    # Normalize the MTF so that it starts at 1
    MTF_1D = MTF_1D / MTF_1D[0]

    # Create the frequency scale for the 1D MTF
    freq_1D = fft.fftfreq(len(intensity), d=(r_values[1] - r_values[0]))

    # Convert from cycles/meter to cycles/mm
    freq_1D_mm = freq_1D #* 1e-3
    
    return freq_1D_mm, MTF_1D

def calc_1D_MTF(r_values,intensity):
    data = fft.fftshift(fft.fft(fft.ifftshift(intensity)))
    #data = fft.fft(fft.ifftshift(intensity))
    dx = r_values[1] - r_values[0]
    
    cx = len(r_values)//2
    dat = abs(data)
    dat /= max(dat)
    freq_1D = fft.fftfreq(len(intensity), d=(r_values[1] - r_values[0]))
    
    # Convert from cycles/meter to cycles/mm
    freq_1D_mm = freq_1D #* 1e-3
    return freq_1D_mm[:cx+1], dat[cx:]

def difflim_mtf(f_stop=4.12, wavelength=550e-6, samples=128):
    extinction = 1 / (wavelength * f_stop)
    normalized_frequency = np.linspace(0, 1, samples)

    mtf = (2 / np.pi) * \
           (np.arccos(normalized_frequency) - normalized_frequency *
            np.sqrt(1 - normalized_frequency ** 2))
    return normalized_frequency * extinction, mtf

if __name__ == "__main__":
    # Generate Airy disk profile
    #r_values, intensity = calculate_airy_disk_profile(f_stop=4.12,r_max=1e-1, num_points=10000)
    r_values, intensity = airy_mtf_improve.airy_disk1d(wavelength=550e-6, f_stop=4.12, r_max=0.1, dx=1e-4, trim=False, normalize=True,trim_zero_index=2, threshold=1e-5)

    # Plot the Airy disk profile
    plt.figure(figsize=(10, 6))
    plt.plot(r_values * 1e3, intensity)
    #plt.xlim(0,15)
    plt.title("Airy Disk Profile")
    plt.xlabel("Radial Distance (μm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True)
    plt.show()

    # Calculate and plot the 1D MTF
    freq_1D_mm, MTF_1D = calc_1D_MTF(r_values, intensity)
    nl = len(MTF_1D)//2
    plt.figure(figsize=(10, 6))
    #plt.plot(freq_1D_mm[:nl], MTF_1D[:nl])
    plt.plot(freq_1D_mm,MTF_1D)
    freq,difflim = difflim_mtf(f_stop=4.12, wavelength=550e-6, samples=128)
    plt.plot(freq,difflim)
    plt.title('1D MTF from 1D Airy Disk Profile')
    plt.xlabel('Spatial Frequency (cycles/mm)')
    plt.ylabel('MTF')
    plt.grid(True)
    #plt.xlim(0, 200)  # Limit to 0-1000 cycles/mm for visibility
    plt.show()
