
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

def calculate_airy_disk_profile(wavelength=550e-9, f_stop=4.12, r_max=1e-3, num_points=100000):
    NA = 1 / (2 * f_stop)
    k = (2 * np.pi / wavelength) * NA
    r_values = np.linspace(0, r_max, num_points)
    intensity = (2 * j1(k * r_values) / (k * r_values))**2
    intensity[0] = 1
    return r_values, intensity

def generate_sinusoidal_pattern(frequency, x_values):
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x_values)

def optimized_convolution_with_sinusoidal(x_values, profile, frequency):
    sinusoidal_pattern = generate_sinusoidal_pattern(frequency, x_values)
    convolved_pattern = np.convolve(profile, sinusoidal_pattern, mode='same')
    convolved_pattern = (convolved_pattern - np.min(convolved_pattern)) / (np.max(convolved_pattern) - np.min(convolved_pattern))
    return convolved_pattern

def calculate_MTF_from_convolution(convolved_pattern, input_contrast):
    central_region = slice(len(convolved_pattern) // 4, 3 * len(convolved_pattern) // 4)
    output_contrast = np.max(convolved_pattern[central_region]) - np.min(convolved_pattern[central_region])
    MTF = output_contrast / input_contrast
    return MTF

if __name__ == "__main__":
    r_values, intensity = calculate_airy_disk_profile()
    first_zero_index = np.where(intensity[1:] < 1e-6)[0][0] + 1
    trimmed_profile = intensity[:first_zero_index]
    trimmed_x_values = r_values[:first_zero_index] * 1e6  # Convert to um

    two_sided_intensity = np.concatenate((np.flip(trimmed_profile[1:]), trimmed_profile))
    two_sided_x_values_um = np.concatenate((-np.flip(trimmed_x_values[1:]), trimmed_x_values))

    cutoff_frequency = 2 * (1 / (2 * 4.12)) / 550e-9 / 1e3  # in cycles/mm
    frequencies = np.linspace(0, cutoff_frequency, 200)
    MTF_values = []

    for freq in frequencies:
        convolved_pattern = optimized_convolution_with_sinusoidal(two_sided_x_values_um / 1000, two_sided_intensity, freq)
        MTF = calculate_MTF_from_convolution(convolved_pattern, 1)
        MTF_values.append(MTF)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, MTF_values)
    plt.title('MTF Curve')
    plt.xlabel('Spatial Frequency (cycles/mm)')
    plt.ylabel('MTF')
    plt.grid(True)
    plt.show()
