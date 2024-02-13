
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.fft import fft, fftfreq

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

def generate_sinusoidal_pattern(frequency, x_values):
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x_values)

def set_xaxis_cycles_or_xlim(ax, frequency, num_cycles=None, xlim=None):
    if num_cycles:
        ax.set_xlim(-num_cycles / frequency*1e3, num_cycles / frequency*1e3)
    elif xlim:
        ax.set_xlim(xlim)

if __name__ == "__main__":
    # Parameters
    frequency = 100  # cycles/mm
    num_cycles = 5  # Number of cycles to show in plot
    xlim = None  # Specific xlim, use None to auto-set based on num_cycles

    # Generate two-sided Airy disk profile
    r_values, intensity = calculate_airy_disk_profile()
    two_sided_intensity = np.concatenate((np.flip(intensity[1:]), intensity))
    #two_sided_x_values_um = np.concatenate((-np.flip(r_values[1:]), r_values)) * 1e6  # Convert to um
    spatial_resolution = r_values[1] - r_values[0]  # Calculate the spatial resolution from the Airy profile
    x_range = np.arange(0, 1e-3, spatial_resolution)
    two_sided_x_values_um = x_range*1e6
    # Generate a sinusoidal pattern
    sinusoidal_pattern = generate_sinusoidal_pattern(frequency, two_sided_x_values_um / 1000)  # Convert to mm

    # Convolve the two-sided Airy disk profile with the sinusoidal pattern
    convolved_pattern = np.convolve(two_sided_intensity, sinusoidal_pattern, mode='same')
    convolved_pattern = (convolved_pattern - np.min(convolved_pattern)) / (np.max(convolved_pattern) - np.min(convolved_pattern))

    # Define a central region to focus on for the MTF calculation
    central_region = slice(len(convolved_pattern) // 4, 3 * len(convolved_pattern) // 4)

    # Calculate the MTF for 440 cycles/mm
    input_contrast = np.max(sinusoidal_pattern) - np.min(sinusoidal_pattern)
    output_contrast = np.max(convolved_pattern[central_region]) - np.min(convolved_pattern[central_region])
    MTF = output_contrast / input_contrast

    print(f"Calculated MTF for {frequency} cycles/mm: {MTF}")

    # Create a 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(two_sided_x_values_um, two_sided_intensity)
    axes[0].set_title('Two-Sided Airy Disk')
    axes[0].set_xlabel('Position (um)')
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True)
    set_xaxis_cycles_or_xlim(axes[0], frequency, num_cycles, xlim)

    axes[1].plot(two_sided_x_values_um, sinusoidal_pattern)
    axes[1].set_title('Input Sinusoidal Pattern')
    axes[1].set_xlabel('Position (um)')
    axes[1].set_ylabel('Intensity')
    axes[1].grid(True)
    set_xaxis_cycles_or_xlim(axes[1], frequency, num_cycles, xlim)

    axes[2].plot(two_sided_x_values_um, convolved_pattern)
    axes[2].set_title('Output Convolved Pattern')
    axes[2].set_xlabel('Position (um)')
    axes[2].set_ylabel('Intensity')
    axes[2].grid(True)
    set_xaxis_cycles_or_xlim(axes[2], frequency, num_cycles, xlim)

    plt.tight_layout()
    plt.show()
