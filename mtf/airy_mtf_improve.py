import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j1  # Bessel function of the first kind
from scipy.interpolate import interp1d

# Define the MTF for a circular aperture
def mtf_circular(frequency, fstop=4.12, wavelength=550e-6):
    """
    Calculate the MTF for a circular aperture.

    :param frequency: spatial frequency (cycles per mm)
    :param diameter: diameter of the aperture (mm)
    :param wavelength: wavelength of the light (mm)
    :return: MTF value at the given spatial frequency
    """
    # Calculate the cutoff frequency
    cutoff_frequency = 1 / (wavelength * fstop)

    # Normalize the spatial frequency
    nu_normalized = frequency / cutoff_frequency

    if nu_normalized <= 1:
        # Calculate the MTF using the formula for a circular aperture
        mtf = (2 / np.pi) * (np.arccos(nu_normalized) - nu_normalized * np.sqrt(1 - nu_normalized**2))
    else:
        mtf = 0

    return mtf

# Define the MTF for a square aperture
def mtf_square(frequency_x, frequency_y, width, wavelength):
    """
    Calculate the MTF for a square aperture.

    :param frequency_x: spatial frequency in the x-direction (cycles per mm)
    :param frequency_y: spatial frequency in the y-direction (cycles per mm)
    :param width: width of the square aperture (mm)
    :param wavelength: wavelength of the light (mm)
    :return: MTF value at the given spatial frequency
    """
    # Calculate the cutoff frequency
    cutoff_frequency = 1 / (wavelength * width)

    # MTF is separable in x and y for a square aperture, so calculate each dimension
    mtf_x = np.sinc(width * frequency_x * wavelength)
    mtf_y = np.sinc(width * frequency_y * wavelength)

    # Combine the x and y MTF values for the 2D MTF
    mtf = np.abs(mtf_x) * np.abs(mtf_y)

    return mtf


def trim_array_two_ends(arr, threshold=1e-6):
    # Find the index of the maximum value
    max_index = np.argmax(arr)

    # Initialize indices for near-zero values
    left_index = max_index
    right_index = max_index

    # Search for the first near-zero value on the left side
    for i in range(max_index, -1, -1):
        if abs(arr[i]) <= threshold:
            left_index = i
            break

    # Search for the first near-zero value on the right side
    for i in range(max_index, len(arr)):
        if abs(arr[i]) <= threshold:
            right_index = i
            break

    # Trim the array
    trimmed_array = arr[left_index:right_index + 1]

    return trimmed_array,left_index,right_index

# Function to calculate the Airy disk profile (unit: mm)
def calculate_airy_disk_profile(wavelength=550e-6, f_stop=4.12, r_max=1, num_points=10000):
    NA = 1 / (2 * f_stop)
    k = (2 * np.pi / wavelength) * NA
    r_values_mm = np.linspace(0, r_max, num_points)
    intensity = (2 * j1(k * r_values_mm) / (k * r_values_mm))**2
    intensity[0] = 1  # Handle the division by zero at the origin
    return r_values_mm, intensity


def airy_disk1d(wavelength=550e-6, f_stop=4.12, r_max=0.1, dx=1e-4, trim=False, normalize=True,trim_zero_index=2, threshold=1e-5):
    NA = 1 / (2 * f_stop)
    k = (2 * np.pi / wavelength) * NA
    r_values_mm = np.arange(0, r_max, dx)
    intensity = (2 * j1(k * r_values_mm) / (k * r_values_mm))**2
    intensity[0] = 1  # Handle the division by zero at the origin
    if trim:
        zero_crossing_indices = np.where(np.diff(np.sign(intensity - threshold)))[0]
        if len(zero_crossing_indices) >= trim_zero_index:
            trim_index = zero_crossing_indices[trim_zero_index - 1] + 1
            intensity_tr = intensity[:trim_index]
            r_values_mm_tr = r_values_mm[:trim_index]
        else:
            print("Not enough zero crossings found, returning untrimmed profile.")
            intensity_tr = intensity
            r_values_mm_tr = r_values_mm
    else:
        intensity_tr = intensity
        r_values_mm_tr = r_values_mm
    intensity2side = np.concatenate((np.flip(intensity_tr[1:]), intensity_tr))
    x_values_mm = np.concatenate((np.flip(-r_values_mm_tr[1:]), r_values_mm_tr))
    if normalize:
        integral = np.sum(intensity2side) #* dx  # Calculate the integral of the intensity profile
        print('sum_intensity2side',integral/dx,dx,integral)
        intensity2side /= integral  # Normalize so the integral is unity
    return x_values_mm, intensity2side

def airy_disk1dold(wavelength=550e-6, f_stop=4.12, r_max=0.1,dx=1e-4,trim=True,threshold=1e-5):
    NA = 1 / (2 * f_stop)
    k = (2 * np.pi / wavelength) * NA
    r_values_mm = np.arange(0, r_max, dx)
    intensity = (2 * j1(k * r_values_mm) / (k * r_values_mm))**2
    intensity[0] = 1  # Handle the division by zero at the origin
    if trim:
        first_zero_index = np.where(intensity[1:] < threshold)[0][0] + 1
        intensity_tr = intensity[:first_zero_index]
        r_values_mm_tr = r_values_mm[:first_zero_index]
    else:
        intensity_tr = intensity
        r_values_mm_tr = r_values_mm
    intensity2side=np.concatenate((np.flip(intensity_tr[1:]), intensity_tr))
    x_values_mm = np.concatenate((np.flip(-r_values_mm_tr[1:]), r_values_mm_tr))
    return x_values_mm, intensity2side

def airy_convolve(x,psf,f_stop,trim = False, wavelength=550e-6):
    dx = abs(x[1]-x[0])
    r_max = 0.15 #abs(x[-1]-x[0])/2
    x_airy,airy=airy_disk1d(wavelength=wavelength,f_stop=f_stop,r_max=r_max,dx=dx)
    norm=0
    for i in range(len(airy)):
        norm+=airy[i]
    airy /= norm
    convolved_pattern = np.convolve(airy, psf, mode='same')
    convolved_x_range_mm = np.linspace(-r_max,r_max, len(convolved_pattern))
    pattern_tr,left_tr,right_tr = trim_array_two_ends(convolved_pattern)
    convolved_x_tr = convolved_x_range_mm[left_tr:right_tr+1]
    #fig=plt.figure()
    #plt.plot(convolved_x_range_mm, convolved_pattern)
    #plt.show()
    if trim:
        return convolved_x_tr,pattern_tr
    else:
        return  convolved_x_range_mm, convolved_pattern

# Function to generate a sinusoidal pattern (frequency in cycles/mm)
def generate_sinusoidal_pattern(frequency, x_values_mm):
    return 0.5 + 0.5 * np.sin(2 * np.pi * frequency * x_values_mm)

# Optimized convolution function with a longer sinusoidal pattern (unit: mm)
def optimized_convolution_with_sinusoidal_fixed_x(profile, frequency, x_range_mm):
    sinusoidal_pattern = generate_sinusoidal_pattern(frequency, x_range_mm)
    convolved_pattern = np.convolve(profile, sinusoidal_pattern, mode='full')
    #convolved_pattern = (convolved_pattern - np.min(convolved_pattern)) / (np.max(convolved_pattern) - np.min(convolved_pattern))
    #convolved_pattern /=np.max(convolved_pattern)
    convolved_x_range_mm = np.linspace(0, x_range_mm[-1], len(convolved_pattern))
    return  convolved_x_range_mm, convolved_pattern

# Function to calculate MTF from convolution
def calculate_MTF_from_convolution(convolved_pattern, input_contrast):
    central_region = slice(len(convolved_pattern) // 4, 3 * len(convolved_pattern) // 4)
    output_contrast = (np.max(convolved_pattern[central_region]) - np.min(convolved_pattern[central_region]))/\
                      (np.max(convolved_pattern[central_region]) + np.min(convolved_pattern[central_region]))
                    
    MTF = output_contrast / input_contrast
    return MTF

# Function to calculate the full MTF curve from 0 to cutoff frequency
def calculate_full_MTFold(profile, cutoff_frequency, x_range_mm, step=10,frequencies=None):
    if frequencies is None:
        frequencies = np.arange(0, cutoff_frequency + step, step)
    MTF_values = []
    for freq in frequencies:
        _, convolved_pattern= optimized_convolution_with_sinusoidal_fixed_x(profile, freq, x_range_mm)
        MTF = calculate_MTF_from_convolution(convolved_pattern, 1)  # Input contrast is 1 for sinusoidal pattern
        if freq in [10,30,60]:
            print('freq=',freq,'MTF=',MTF)
        MTF_values.append(MTF)
    MTF_values[0] = 1  # Make sure MTF at zero frequency is 1
    return frequencies, MTF_values
def generate_axisymmetric_2d(array_1d, plot=False):
    """
    Generates a 2D axisymmetric array from a 1D numpy array and optionally plots it.
    
    Parameters:
    - array_1d: Input 1D numpy array.
    - plot: Boolean flag to control plotting of the 2D array. Default is False.
    
    Returns:
    - A 2D numpy array that is axisymmetric with respect to the maximum value of the input 1D array.
    """
    # Find the index of the maximum value and extract the left half including the maximum value
    idx_max = np.argmax(array_1d)
    left_half = array_1d[:idx_max + 1]
    
    # Create a linear interpolation function
    distances = np.arange(idx_max + 1)[::-1]  # Distances from the max value in terms of index
    interpolation = interp1d(distances, left_half, fill_value="extrapolate", bounds_error=False)
    
    # Generate a 2D grid
    size = 2 * len(left_half) - 1  # The side length of the square 2D grid
    x, y = np.indices((size, size))
    center = len(left_half) - 1
    dist_to_center = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Use the interpolation to assign values based on distance to the 2D grid
    axisymmetric_2d = interpolation(dist_to_center)
    
    # Optionally plot the 2D axisymmetric array
    if plot:
        plt.figure(figsize=(8, 8))
        #plt.plot(array_1d)
        #plt.plot(left_half)
        plt.imshow(axisymmetric_2d, cmap='viridis')
        plt.colorbar()
        plt.title('2D Axisymmetric Array via Linear Interpolation')
        plt.show()
    
    return axisymmetric_2d
from scipy import fft

def fftrange(n, dtype=None):
    """FFT-aligned coordinate grid for n samples."""
    # return np.arange(-n//2, -n//2+n, dtype=dtype)
    return np.arange(-(n//2), -(n//2)+n, dtype=dtype)

def calc_2D_MTF(intensity,dx):
    data = fft.fftshift(fft.fft2(fft.ifftshift(intensity)))
    #data = fft.fft(fft.ifftshift(intensity))
    cy, cx = (int(np.floor(s / 2)) for s in data.shape)
    #cx = len(x)//2
    dat = abs(data)
    #dat /= max(dat[cx])
    dat /= dat[cy,cx]
    #freq_1D = fft.fftfreq(data.shape[0], d=dx)
    df = 1./(data.shape[0]*dx)
    print('shape',data.shape[0],'dx',dx,'df',df)
    #freq_1D = np.linspace(0.)
    s=data.shape[0]
    freq_1D= fftrange(s) * df
    # Convert from cycles/meter to cycles/mm
    freq_1D_mm = freq_1D #* 1e-3
    #return freq_1D_mm[:cx], dat[cx:]
    return freq_1D_mm[cx:], dat[cx,cx:]

def calculate_full_MTF(profile, cutoff_frequency, x_range_mm, step=10,frequencies=None):
    #if frequencies is None:
    #    frequencies = np.arange(0, cutoff_frequency + step, step)
    psf2d = generate_axisymmetric_2d(profile,plot=False)
    dx = x_range_mm[1]-x_range_mm[0]
    freqs, MTFs = calc_2D_MTF(psf2d,dx)
    if frequencies is None:
        return freqs, MTFs
    #frequencies=freqs
    #MTF_values=MTFs
    if frequencies is not None:
        interpolation = interp1d(freqs, MTFs)#, fill_value="extrapolate", bounds_error=False)
        MTF_values=interpolation(frequencies)
        print('########FREQs',freqs[:5],MTFs[:5])
        print('########FRLSs',frequencies,MTF_values)
        return frequencies, MTF_values
# Update the 3-panel plot function to take into account spatial resolution for sinusoidal pattern
def plot_three_panel_figure(airy_profile, x_values_mm, x_range_mm, input_frequency=100, xlim_um=None, N_cycles=None):
    # Generate the sinusoidal pattern with the same spatial resolution as the Airy disk profile
    sinusoidal_pattern = generate_sinusoidal_pattern(input_frequency, x_range_mm)
    
    # Perform the convolution
    convolved_x_range_mm, convolved_pattern= optimized_convolution_with_sinusoidal_fixed_x(airy_profile, input_frequency, x_range_mm)
    MTF=calculate_MTF_from_convolution(convolved_pattern, 1) 
    print('MTF',MTF,'freq',input_frequency)
    # Convert mm to um for plotting
    x_values_um = x_values_mm * 1e3
    x_range_um = x_range_mm * 1e3
    convolved_x_range_um = convolved_x_range_mm * 1e3
    
    # Calculate xlim based on N_cycles
    if N_cycles is not None:
        xlim_um = [0, N_cycles / input_frequency * 1e3]  # Convert cycles/mm to cycles/um
    
    # Create the 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].plot(x_values_um, airy_profile)
    axes[0].set_title('Airy Disk Profile')
    axes[0].set_xlabel('Spatial Position (um)')
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True)
    
    axes[1].plot(x_range_um, sinusoidal_pattern)
    axes[1].set_title('Input Sinusoidal Pattern')
    axes[1].set_xlabel('Spatial Position (um)')
    
    axes[2].plot(x_range_um, sinusoidal_pattern)
    axes[2].plot(convolved_x_range_um, convolved_pattern)
    axes[2].set_title('Output Convolved Pattern')
    axes[2].set_xlabel('Spatial Position (um)')
    
    for ax in axes[1:]:
        ax.grid(True)
        if xlim_um:
            ax.set_xlim(xlim_um)
    
    plt.show()

def plot_MTF(psf_x,psf,ax):
    r_values_mm = psf_x
    intensity   = psf
    norm=0.
    for i in range(len(intensity)):
        norm +=intensity[i]
    intensity /= norm

    spatial_resolution = r_values_mm[1] - r_values_mm[0]  # Calculate the spatial resolution from the Airy profile
    x_range_mm = np.arange(0, 1, spatial_resolution)  # Here 1 is 1 mm

    # Calculate the cutoff frequency (in cycles/mm)
    cutoff_frequency = 200 #1 / (4.12 * 550e-6)
# Calculate the full MTF curve
    frequencies, MTF_values = calculate_full_MTF(intensity, int(cutoff_frequency), x_range_mm)

# Make sure the MTF at zero frequency is 1
    MTF_values[0] = 1

# Plot the MTF curve with the final corrected cutoff frequency and MTF(0) = 1
    ax.plot(frequencies, MTF_values, marker='o')
    ax.set_ylim(0,1.05)
    #ax.title('MTF Curve with Final Corrected Cutoff Frequency and MTF(0) = 1')
    ax.set_xlabel('Spatial Frequency (cycles/mm)')
    ax.set_ylabel('MTF')
    ax.grid(True)
    #plt.show()


"""
# Generate two-sided Airy disk profile (unit: mm)
r_values_mm, intensity = calculate_airy_disk_profile(r_max=1)
first_zero_index = np.where(intensity[1:] < 1e-6)[0][0] + 1
trimmed_profile = intensity[:first_zero_index]
trimmed_r_value_mm = r_values_mm[:first_zero_index]
two_sided_intensity = np.concatenate((np.flip(trimmed_profile[1:]), trimmed_profile))
norm=0.
for i in range(len(two_sided_intensity)):
    norm +=two_sided_intensity[i]

two_sided_intensity/=norm

x_values_mm = np.concatenate((np.flip(trimmed_r_value_mm[1:]),trimmed_r_value_mm))
# Correct the x_range_mm to actually extend to 1 mm (unit: mm)
spatial_resolution = r_values_mm[1] - r_values_mm[0]  # Calculate the spatial resolution from the Airy profile
x_range_mm = np.arange(0, 1, spatial_resolution)  # Here 1 is 1 mm

# Calculate the cutoff frequency (in cycles/mm)
cutoff_frequency = 1 / (4.12 * 550e-6)



# Test the 3-panel plot function with 40 lp/mm and xlim to show 5 cycles
plot_three_panel_figure(two_sided_intensity, x_values_mm, x_range_mm, input_frequency=100, N_cycles=5)



# Calculate the full MTF curve
frequencies, MTF_values = calculate_full_MTF(two_sided_intensity, int(cutoff_frequency), x_range_mm)

# Make sure the MTF at zero frequency is 1
MTF_values[0] = 1

# Plot the MTF curve with the final corrected cutoff frequency and MTF(0) = 1
plt.figure(figsize=(10, 6))
plt.plot(frequencies, MTF_values, marker='o')
plt.title('MTF Curve with Final Corrected Cutoff Frequency and MTF(0) = 1')
plt.xlabel('Spatial Frequency (cycles/mm)')
plt.ylabel('MTF')
plt.grid(True)
plt.show()
"""
