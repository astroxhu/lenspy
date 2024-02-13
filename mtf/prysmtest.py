from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import regular_polygon

from matplotlib import pyplot as plt

efl = 500
fno = 4.12

x, y = make_xy_grid(100, diameter=efl/fno)
dx = x[0,1]-x[0,0]
r, t = cart_to_polar(x, y)
radius = efl/fno/2
rho = r / radius
n_sides = 10000

aperture = regular_polygon(n_sides, radius, x, y)

plt.imshow(aperture, origin='lower')

from prysm.polynomials import zernike_nm
from prysm.propagation import Wavefront
wvl = 0.55 # mid visible band, um

wfe_nm_rms = wvl/14*1e3 # nm, 3/4 of a wave, 1e3 = um to nm
mode = zernike_nm(4, 0, rho, t)
opd = mode * wfe_nm_rms
opd = None
pup = Wavefront.from_amp_and_phase(aperture, opd, wvl, dx)
coherent_psf = pup.focus(efl, Q=16)
from prysm.otf import mtf_from_psf, diffraction_limited_mtf, transform_psf
psf = coherent_psf.intensity
mtf = mtf_from_psf(psf)
print('psf shape', psf.data.shape)
fx, _ = mtf.slices().x
#fig, ax = mtf.slices().plot(['x', 'y', 'azavg'], xlim=(0,200))
fig, ax = mtf.slices().plot(['azavg'], xlim=(0,200))

difflim = diffraction_limited_mtf(fno, wvl, fx)

ax.plot(fx, difflim, ls=':', c='k', alpha=1, zorder=1,lw=2)
ax.set(xlabel='Spatial frequency, cy/mm', ylabel='MTF')

plt.show()

import airy2d_mtf as a2
import numpy as np
r_max=0.113
res=1600
dx=2*r_max/res
print('psf resolution',dx)
norm=np.max(psf.data)
x,y,airy=a2.generate_2D_airy_disk(wavelength=550e-6, f_stop=fno, r_max=r_max, grid_size=res)
from prysm._richdata import RichData
print('airy shape',airy.shape)
psf2=RichData(data=norm*airy,dx=1000*dx,wavelength=None)

#print('psf2 shape', psf.data.shape)
mtf2 = mtf_from_psf(psf2)

fx, _ = mtf2.slices().x
#fig, ax = mtf.slices().plot(['x', 'y', 'azavg'], xlim=(0,200))
fig, ax = mtf.slices().plot(['x'], xlim=(0,1000))

mtf2.slices().plot(['x'], xlim=(0,1000),ax=ax)
difflim = diffraction_limited_mtf(fno, wvl, fx)

ax.plot(fx, difflim, ls=':', c='k', alpha=0.75, zorder=1)
ax.set(xlabel='Spatial frequency, cy/mm', ylabel='MTF')
if True:
    freq_1D_mm, MTF_1D = a2.calc_2D_MTF(airy,dx)
    nl = len(MTF_1D)//2
#plt.plot(freq_1D_mm[:nl], MTF_1D[:nl])
    #first0=np.where(MTF_1D<1e-3)[0][0]
    wavelength=550e-6
    f_stop=4.12
    extinction = 1 / (wavelength * f_stop)
    #print(first0)
    #freq=np.linspace(0,first0,first0)
    #freq*=extinction/first0
    ax.plot(freq_1D_mm,MTF_1D[:],ls='--',label='mine')
    ax.legend()
plt.show()
fig=plt.figure()
ax=plt.subplot(111)
psf.slices().plot(['x'],ax=ax)
psf2.slices().plot(['x'],ax=ax)

plt.show()
