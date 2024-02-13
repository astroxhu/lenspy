import numpy as np
import matplotlib.pyplot as plt

import airy_mtf_improve

freq=10
dx=1e-5
x_range,psf = airy_mtf_improve.airy_disk1d(dx=dx,normalize=True)

x_range0=np.arange(0,1,dx)
sin_pattern=airy_mtf_improve.generate_sinusoidal_pattern(freq, x_range0)
x_range_con,convolved_pattern=airy_mtf_improve.optimized_convolution_with_sinusoidal_fixed_x(psf, freq, x_range0)

fig=plt.figure()
ax=plt.subplot(111)
ax.plot(x_range0,sin_pattern)
ax.plot(x_range_con,convolved_pattern)
print('mtf',airy_mtf_improve.calculate_MTF_from_convolution(convolved_pattern, 1))
plt.show()
