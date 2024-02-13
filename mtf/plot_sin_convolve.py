import airy_mtf_improve
import numpy as np

dx=1e-5
r_max=0.1
x_airy,airy=airy_mtf_improve.airy_disk1d(wavelength=550e-6,f_stop=4.12,r_max=r_max,dx=dx)
airy /=sum(airy)
x_range_mm = np.arange(0, 1, dx)  # Here 1 is 1 mm
airy_mtf_improve.plot_three_panel_figure(airy, x_airy, x_range_mm, input_frequency=100, N_cycles=5)
