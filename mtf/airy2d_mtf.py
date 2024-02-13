import numpy as np
import matplotlib.pyplot as plt
import airy_mtf_improve
from scipy.signal import convolve2d
#import numpy.fft as fft
from scipy import fft

# Create 2D Airy disk profile
def generate_2D_airy_disk(wavelength=550e-6, f_stop=4.12, r_max=1, grid_size=1000):
    r_values_mm, intensity_1D =  airy_mtf_improve.calculate_airy_disk_profile(wavelength, f_stop, r_max, grid_size)
    x = np.linspace(-r_max, r_max, grid_size)
    y = np.linspace(-r_max, r_max, grid_size)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    intensity_2D = np.interp(r, r_values_mm, intensity_1D)
    return x, y, np.array(intensity_2D)

def circle2dfunc(wavelength=550e-6, f_stop=4.12, r_max=1, grid_size=100):
    x = np.linspace(-r_max, r_max, grid_size)
    y = np.linspace(-r_max, r_max, grid_size)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    r0=1.22*wavelength*f_stop
    # Create a 2D map where the value is 1 if r < r0 and 0 if r > r0
    map_2d = np.where(r < r0, 1.0, 0.0)

    return map_2d

    
# Generate adjusted 2D sinusoidal pattern
def generate_2D_sinusoidal_pattern(frequency=10, phase=0, amplitude=0.5, dx=0.01, extent_x=0.5, extent_y=0.2):
    x = np.arange(-extent_x, extent_x+dx, dx)
    y = np.arange(-extent_y, extent_y+dx, dx)
    x, y = np.meshgrid(x, y)
    sinusoidal_pattern = amplitude*(1 + np.sin(2 * np.pi * frequency * x + phase))
    #sinusoidal_pattern = amplitude*(1+np.sign(np.sin(2 * np.pi * frequency * x + phase)))
    return x, y, sinusoidal_pattern

# Function to perform 2D convolution and crop the result to the original size
def convolve_and_crop(image1, image2, mode='full'):
    convolved_image = convolve2d(image1, image2, mode=mode)
    crop_size_x = (convolved_image.shape[0] - image1.shape[0]) // 2
    crop_size_y = (convolved_image.shape[1] - image1.shape[1]) // 2
    print('shape',convolved_image.shape[0],image1.shape[0],convolved_image.shape[1],image1.shape[1])
    cropped_image = convolved_image[crop_size_x:-crop_size_x, crop_size_y:-crop_size_y]
    return convolved_image

# Function to calculate the contrast of a 1D signal
def calculate_contrast(signal):
    leng = len(signal)//4
    max_val = np.max(signal[leng:leng*3])
    min_val = np.min(signal[leng:leng*3])
    print('max,min',max_val,min_val)
    contrast = (max_val - min_val) / (max_val + min_val)
    return contrast

# Function to create a 3-panel plot
def plot_three_panel(x,y,original_pattern, convolved_pattern, slice_1D_original, slice_1D_convolved):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Left Panel: Original sinusoidal pattern
    #ax1.imshow(original_pattern, cmap='gray', extent=[x[0], x[-1], y[0], y[-1]])
    ax1.pcolormesh(x,y,original_pattern, cmap='gray')
    ax1.set_title('Original Sinusoidal Pattern')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.axis('equal')

    # Center Panel: Convolved sinusoidal pattern
    #ax2.imshow(convolved_pattern, cmap='gray')
    ax2.pcolormesh(x,y,convolved_pattern, cmap='gray')
    ax2.set_title('Convolved Sinusoidal Pattern')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.axis('equal')

    # Right Panel: 1D slice of both the original and convolved sinusoidal patterns
    ax3.plot(x[0],slice_1D_original, label='Original')
    ax3.plot(x[0],slice_1D_convolved, label='Convolved')
    #ax3.set_title(f'1D Slice along x-axis at y=0\n(MTF = {mtf_value:.4f})')
    ax3.set_xlabel('Pixel Index')
    ax3.set_ylabel('Intensity')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

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
    #freq_1D = np.linspace(0.)
    s=data.shape[0]
    freq_1D= fftrange(s) * df
    # Convert from cycles/meter to cycles/mm
    freq_1D_mm = freq_1D #* 1e-3
    #return freq_1D_mm[:cx], dat[cx:]
    return freq_1D_mm[cx:], dat[cx,cx:]


if __name__ == "__main__":
    rmax=0.113
    res=160
    dx=2*rmax/res
# Generate a high-resolution 2D Airy disk with the specified size
    x, y, airy_2D = generate_2D_airy_disk(wavelength=550e-6, r_max=rmax, f_stop=4.12,grid_size=res)
    airy_2D /= airy_2D.sum()  # Normalize

    plt.imshow(airy_2D, extent=[-rmax,rmax,-rmax,rmax])
    plt.show()

# Calculate and plot the 1D MTF
    freq_1D_mm, MTF_1D = calc_2D_MTF(airy_2D,dx)
    nl = len(MTF_1D)//2
    plt.figure(figsize=(10, 6))
#plt.plot(freq_1D_mm[:nl], MTF_1D[:nl])
    #first0=np.where(MTF_1D<1e-2)[0][0]
    wavelength=550e-6
    f_stop=4.12
    extinction = 1 / (wavelength * f_stop)
    #print(first0)
    #freq=np.linspace(0,first0,first0)
    #freq*=extinction/first0
    plt.plot(MTF_1D[:])
    plt.show()

    circle2d= circle2dfunc(wavelength=650e-6, f_stop=4.12, r_max=rmax, grid_size=res)
#x, y, circle2d = generate_2D_airy_disk(r_max=0.006, f_stop=4.12,grid_size=60)
    circle2d/= circle2d.sum()
    plt.imshow(circle2d, extent=[-rmax,rmax,-rmax,rmax])
    plt.show()
    #airy_2D = convolve_and_crop(airy_2D, circle2d,mode='same')
    plt.imshow(airy_2D, extent=[-rmax,rmax,-rmax,rmax])
    plt.show()
    dx=x[0,1]-x[0,0]
    print('dx',dx)
# Generate adjusted 2D sinusoidal pattern
    x_sin, y_sin, sinusoidal_pattern = generate_2D_sinusoidal_pattern(frequency=10,dx=dx,extent_x=0.6, extent_y=0.2)
    slice_1D_original = sinusoidal_pattern[3,:]
# Convolve high-resolution 2D Airy disk with adjusted 2D sinusoidal pattern
    convolved_pattern = convolve_and_crop(sinusoidal_pattern,airy_2D,mode='same')

#convolved_pattern = sinusoidal_pattern
    mid_index = convolved_pattern.shape[0] // 2
    slice_1D_convolved = convolved_pattern[mid_index,:]
# Calculate the MTF for the high-resolution slice
    mtf = calculate_contrast(slice_1D_convolved)

    print("MTF",mtf)
    plot_three_panel(x_sin,y_sin,sinusoidal_pattern, convolved_pattern, slice_1D_original, slice_1D_convolved)
# Take a slice along the x-axis at y=0 from the high-resolution convolved image

