
import sys
import os
sys.path.append(os.path.abspath(os.pardir))
sys.path.append("/home/uiao/lenspy/mtf")
from lenspy import *
import matplotlib.pyplot as plt
import numpy as np
import optictxt
digtype=np.float32

#filename='rf100300airdiam.txt'
#filename='rf150_2airdiam.txt'
#filename='z2450airdiam.txt'
#filename='ef640_3jpair.txt'
#filename='ef640_4ajpairdiam.txt'
#filename='Nikkor640eairdiam.txt'
#filename='Nikkor856eairdiam.txt'
#filename='z5018_9airdiam.txt'
#filename='z58noct_1airdiam.txt'
#filename='sony1224GMair.txt'
#filename='sigma540jpairdiam.txt'
filename='sigma540bjpairdiam.txt'
#filename='sigma640jpairdiam.txt'
#filename='sigma856jpairdiam.txt'
#filename='sigma856ajpairdiam.txt'
#filename='rf200800jpairdiam.txt'
f_leng=488
f_stop=4.12

###### incident rays #####
Nray=100
npos=5
raywidth=0.5
x0=-3130e3
h0=21.5/f_leng*x0
x_in=500
hray=f_leng/f_stop
dimfac=0.5
ray_end= 410.92 #313.27 #152.15 #160.766 #92.32 #410.92 #179.36
#### evaluation #####
evaluate=True
full_mtf=True
Neval=1000
weval=1.0
#### element location ####
location=0
lenslw=1.
lenscolor='k'
def getpsf(raylist,xfocus,N0=100,N=100,width=1.0):
  print('xfocus',xfocus)
  Nray=len(raylist)
  yarr=np.zeros(Nray)
  x0=xfocus
  for i in range(Nray):
    yarr[i]=-raylist[i].A/raylist[i].B*x0-raylist[i].C/raylist[i].B

  ymax=max(yarr)
  ymin=min(yarr)
  print("psf's ymax,ymin",ymax,ymin)
  wid0=ymax-ymin
  pix=(ymax-ymin)/N0
  xradf=np.linspace(ymin,ymax,N+1)
  xradc=(xradf[1:]+xradf[:-1])/2 
  count=np.zeros(N)
  for i in range(Nray):
    j=int(np.floor((yarr[i]-ymin)/pix))
    #print(i,yarr[i],ymin,pix,j)
    if j>=0 and j<N0:
      count[j]+=1
  idxc=np.argmax(count)
  yc = yarr[idxc]
  ymax = width*wid0/2+yc
  ymin =-width*wid0/2+yc
  #### repeat for new grids
  print('ymax,ymin,yc',ymax,ymin,yc)
  ymax=yc+0.1
  ymin=yc-0.1
  pix=(ymax-ymin)/N
  xradf=np.linspace(ymin,ymax,N+1)
  #xradf=np.linspace(-1,1,N+1)
  xradc=(xradf[1:]+xradf[:-1])/2
  count=np.zeros(N)
  for i in range(Nray):
    j=int(np.floor((yarr[i]-ymin)/pix))
    if j>=0 and j<N:
      count[j]+=1
  return [xradc,count]

def convertpar(surf):
  nd_air = 1.00027717
  vd_air = 89.30
  n_surf=len(surf)
  n_element=n_surf
  i_lens=0
  lens=[dict() for x in range(n_element)]
  for i in  range(1,n_surf+1):
    #print('i',i, surf[i])
    if surf[i]['nd']>1.01:
      lens[i_lens]['num']=i_lens+1
    
      xc=0.
      #print(i_lens,i)
      lens[i_lens]['curv_L']=1.0/digtype(surf[i]['r'])
      lens[i_lens]['curv_R']=1.0/digtype(surf[i+1]['r'])
      if surf[i]['r']>100000.:
        lens[i_lens]['curv_L']=digtype(0.)
      if surf[i+1]['r']>100000.:
        lens[i_lens]['curv_R']=digtype(0.)
      lens[i_lens]['type_L']=surf[i]['type']
      lens[i_lens]['type_R']=surf[i+1]['type']
      if i>1:
        lens[i_lens]['n_L']=digtype(surf[i-1]['nd'])
      else:
        lens[i_lens]['n_L']=nd_air
      lens[i_lens]['n_in']=digtype(surf[i]['nd'])
      lens[i_lens]['n_R']=digtype(surf[i+1]['nd'])
      lens[i_lens]['diam_L']=digtype(surf[i]['diam'])
      lens[i_lens]['diam_R']=digtype(surf[i+1]['diam'])

      for ii in range(1,i):
        #print(ii,surf[ii]['r'])
        xc+=digtype(surf[ii]['d'])
      lens[i_lens]['xc']=digtype(xc)
      lens[i_lens]['thick']=digtype(surf[i]['d'])

      if surf[i]['type']=='EVENASPH' or surf[i]['type']=='STANDARD':
        lens[i_lens]['parm_L']=digtype(surf[i]['parm'])
        lens[i_lens]['parm_R']=digtype(surf[i+1]['parm'])
        lens[i_lens]['conic_L']=digtype(surf[i]['conic'])
        lens[i_lens]['conic_R']=digtype(surf[i+1]['conic'])
      i_lens+=1
  filtered_lens = [d for d in lens if d]
  filtered_lens[-1]['BF']=digtype(surf[n_surf]['d'])
  print('Back Focus',filtered_lens[-1]['BF'],digtype(surf[n_surf]['d']))
  return filtered_lens
# Use the function to parse the file
#data = optictxt.parse_optical_file("rf100300airdiam.txt",loc=2)
#data = optictxt.parse_optical_file("rf5018_6airdiam.txt")
#data = optictxt.parse_optical_file("rf220_1airdiam.txt")
data = optictxt.parse_optical_file(filename,loc=location)
#print(data)
for i in range(1,len(data)+1):
  print(i,data[i],'\n')
lens_params=convertpar(data)
lens_params[-1]['xc']+=0.08
if not ray_end:
    ray_end=lens_params[-1]['xc']+lens_params[-1]['thick']+1.
    if lens_params[-1]['BF']:
         ray_end=lens_params[-1]['xc']+lens_params[-1]['thick']+lens_params[-1]['BF']
#lens_params=readzmx_asph("mobile-iPhone_USP20170299845.zmx")
#lens_params=readzmx_asph("DoubleGauss_Wakiyama_USP4448497.zmx")
print('lens_params',lens_params)
####### edit some lens #######
#lens_params[4]['n_R']=1.514
#lens_params[5]['n_L']=1.514
for i in range(0*len(lens_params)):
  print(i,lens_params[i]['n_L'],lens_params[i]['n_in'],lens_params[i]['n_R'],'curv',lens_params[i]['curv_L'],lens_params[i]['curv_R'],'pos',lens_params[i]['xc'],lens_params[i]['thick'])
#lens_params[-2]['parm_L']= [0.0,-5.93134E-05,-2.83344E-06,1.13683E-7,-2.41213E-09,2.81858E-11,-1.74792E-13,4.56347E-16]
#print('last lens',lens_params[-2])
lens=lensnew(lens_params)

fig=plt.figure(figsize=(14,8.))
if evaluate:
  ax1=plt.subplot(211)
else:
  ax1=plt.subplot(111)
ax1.set_aspect('equal')
#ax1.set_xlim(-2,7)
#ax1.set_ylim(-4,4)
ax1.set_xlim(-25,400)
ax1.set_ylim(-80,80)
ax1.set_xlim(-25,lens_params[-1]['xc']*1.5)
ax1.set_ylim(-lens_params[0]['diam_L']/1.6,lens_params[0]['diam_L']/1.6)
#ax1.set_xlim(39.3,39.34)
#ax1.set_ylim(5.85,5.88)
#Nray=10
lens.assemble(ax1,lw=lenslw,lenscolor=lenscolor)
for i in range(npos-1,-1,-1):
    Ray = [ray2d(0,0,0,0,0,color='C'+str(i+2)) for j in range(Nray)]
    if npos>1:
        pos_step=h0/(npos-1)
        dim=dimfac**(i/(npos-1))
    else:
        pos_step=h0
        dim=1

    Ray = gen_optics().ray_in(x0,(-i)*pos_step,x_in,hray*dim,Ray)
    [rayout,rayout1]=lens.raytrace(ax1,Ray,raywidth=raywidth,rayend=ray_end)
#print('lenRay',len(Ray))
#Ray[:Nray]=gen_optics().ray_in(x0,-h0,x_in,hray,Ray[:Nray])
#Ray[Nray:2*Nray]=gen_optics().ray_in(x0,0*h0,x_in,hray,Ray[Nray:2*Nray])
#Ray[2*Nray:]=gen_optics().ray_in(x0,-0.5*h0,x_in,hray,Ray[2*Nray:])
#Ray=gen_optics().ray_in(x0,h0,1,1,Ray)



ax1.vlines(ray_end,-27,27,colors=lenscolor,lw=lenslw)
#print('rayout',rayout[0].xs,rayout[0].xe)
ydist=np.zeros(Nray)

xfocus= 164.4827880332731
xfocus=164.575
xfocus=125.00
xfocus=75.8015+11.438
xfocus=ray_end
#[xfocus,ymax,ymin]=lens.findfocus(rayout,step=2,x0=ray_end*0.9,x1=175.)
#dy=ymax-ymin
#print('focus point', xfocus,ymax,ymin,dy)
#ax1.scatter([xfocus,xfocus],[ymin,ymax])
#ax1.vlines(xfocus,-100,100)
#Ray=[ray2d(0,0,0,0,0) for i in range(Nray)]
#print('lenRay',len(Ray))
#Ray=gen_optics().ray_in(x0,0,0.3,1,Ray)
#Ray=gen_optics().ray_in(x0,200,0.3,50,Ray)

if evaluate:
  #lens.raytrace(ax1,Ray)
  ax2=plt.subplot(245)
  #gen_optics().plotray(ax2,100,rayout[0])
  #ax2.set_xlim(-25,200)
  #ax2.set_ylim(-60,60)
  #ax2.set_aspect('equal')
  n=Neval
  [x,count]=getpsf(rayout,xfocus,N=n,width=weval)
  count=count/np.sum(count)
  #count*=0
  #count[n//2]=1
  ft_c = np.abs(np.fft.fft(count))
  d=x[1]-x[0]
  print('psf resolution',d)
  freq = np.fft.fftfreq(n, d)
  half_n = n//2
  ft_c = ft_c[:half_n]
  freq = np.abs(freq[:half_n])  # only non-negative frequencies
  import airy_mtf_improve
  x_convolve,psf_convolve=airy_mtf_improve.airy_convolve(x,count,f_stop)
  #airy_mtf_improve.plot_MTF(x_convolve,psf_convolve,ax2)
  airy_mtf_improve.plot_MTF(x,count,ax2)
  #ax2.plot(freq,ft_c)

  ax3=plt.subplot(246)
  ax3.plot((x-x[np.argmax(count)])*1e3,count)
  ax3.set_xlim(-10,10)
  ax3.set_xlabel(r'$\mu m$')
  ax4=plt.subplot(247)
  airy_mtf_improve.plot_MTF(x_convolve,psf_convolve,ax4)
  nfreq=200
  freq=np.linspace(0,1000,nfreq)
  mtf_diff=np.zeros(nfreq)
  for i in range(nfreq):
      mtf_diff[i] = airy_mtf_improve.mtf_circular(freq[i])
  ax4.plot(freq,mtf_diff)

  ax3=plt.subplot(248)
  ax3.plot((x_convolve-x_convolve[np.argmax(psf_convolve)])*1e3,psf_convolve)
  ax3.set_xlim(-10,10)
  ax3.set_xlabel(r'$\mu m$')
#print(x,count)
plt.subplots_adjust(left=0.07,right=0.98)
plt.show()


if full_mtf:
    Nray=100
    npos=4
    hmax=h0 #20./500.*x0
    lw=0.8
    ms=2
    mtf10=np.zeros(npos)
    mtf30=np.zeros(npos)
    mtf60=np.zeros(npos)
    mtf10g=np.zeros(npos)
    mtf30g=np.zeros(npos)
    mtf60g=np.zeros(npos)
    x_pos=np.zeros(npos)
    for i in range(npos):
        Ray = [ray2d(0,0,0,0,0,color='C'+str(i+2)) for j in range(Nray)]
        if npos>1:
            pos_step=hmax/(npos-1)
            dim=dimfac**(i/(npos-1))
        else:
            pos_step=h0
            dim=1

        Ray = gen_optics().ray_in(x0,(i)*pos_step,x_in,hray*dim,Ray)
        [rayout,rayout1]=lens.raytrace(ax1,Ray,raywidth=raywidth,rayend=ray_end)
        [x,count]=getpsf(rayout,xfocus,N=n,width=weval)
        count=count/np.sum(count)
        count*=0
        count[n//2]=1
        d=x[1]-x[0]
        x_pos[i] = x[np.argmax(count)]
        posfac=1/np.sqrt(1-(x_pos[i]*20)**2/f_leng**2)
        print('######posfac',posfac,x_pos[i])
        import airy_mtf_improve
        x_range_mm = np.arange(0, 1, d)
        freqlist=np.array([10,20,40,80])
        frequencies, MTF_values_geo = airy_mtf_improve.calculate_full_MTF(count, 200, x_range_mm,frequencies=freqlist)
        x_convolve,psf_convolve=airy_mtf_improve.airy_convolve(x,count,f_stop*posfac,wavelength=550e-6)
        print('x_convolve',x_convolve,x_convolve[1]-x_convolve[0],x_convolve[-1]-x_convolve[0],len(x_convolve))
        frequencies, MTF_values = airy_mtf_improve.calculate_full_MTF(psf_convolve, 200, x_range_mm,frequencies=freqlist)
        mtf10g[i] = MTF_values_geo[0]
        mtf10[i]  = MTF_values[0]
        mtf30g[i] = MTF_values_geo[1]
        mtf30[i]  = MTF_values[1]
        mtf60g[i] = MTF_values_geo[2]
        mtf60[i]  = MTF_values[2]

    fig=plt.figure(figsize=(12.5,6))
    ax1=plt.subplot(121)
    ax1.plot(x_pos,mtf10g,marker='o',label=str(freqlist[0])+' lp/mm',lw=lw,ms=ms)
    ax1.plot(x_pos,mtf30g,marker='o',label=str(freqlist[1])+' lp/mm',lw=lw,ms=ms)
    ax1.plot(x_pos,mtf60g,marker='o',label=str(freqlist[2])+' lp/mm',lw=lw,ms=ms)
    ax1.set_title('Geometrical MTF')
    ax1.set_ylabel('CONTRAST')
    ax1.set_xlabel('IMAGE HEIGHT mm')
    ax1.set_ylim(0,1.05)
    ax1.set_xlim(0,21.5)
    ytick_locations = np.arange(0.0, 1.1, 0.1)
    ax1.set_yticks(ytick_locations)
    ax1.set_yticklabels([str(round(tick, 1)) for tick in ytick_locations])
    xtick_locations = np.arange(0.0, 21.5, 5)
    ax1.set_xticks(xtick_locations)
    ax1.set_xticklabels([str(round(tick, 1)) for tick in xtick_locations])
    ax1.legend(loc=3,fontsize='small')
    ax1.grid(True)
    ax2=plt.subplot(122)
    ax2.plot(x_pos,mtf10,marker='o',label=str(freqlist[0])+' lp/mm',lw=lw,ms=ms)
    ax2.plot(x_pos,mtf30,marker='o',label=str(freqlist[1])+' lp/mm',lw=lw,ms=ms)
    ax2.plot(x_pos,mtf60,marker='o',label=str(freqlist[2])+' lp/mm',lw=lw,ms=ms)
    ax2.set_title('Diffraction MTF')
    ax2.set_ylabel('CONTRAST')
    ax2.set_xlabel('IMAGE HEIGHT mm')
    ax2.set_ylim(0,1.05)
    #ax2.set_xlim(0,21.5)
    ytick_locations = np.arange(0.0, 1.1, 0.1)
    ax2.set_yticks(ytick_locations)
    ax2.set_yticklabels([str(round(tick, 1)) for tick in ytick_locations])
    xtick_locations = np.arange(0.0, 21.5, 5)
    ax2.set_xticks(xtick_locations)
    ax2.set_xticklabels([str(round(tick, 1)) for tick in xtick_locations])
    ax2.legend(loc=3,fontsize='small')
    ax2.grid(True)
    plt.subplots_adjust(left=0.07,right=0.95)
    plt.show()
