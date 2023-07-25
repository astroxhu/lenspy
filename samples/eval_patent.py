
import sys
import os
sys.path.append(os.path.abspath(os.pardir))
from lenspy import *
import matplotlib.pyplot as plt
import numpy as np
import optictxt
digtype=np.float32

filename='rf100300airdiam.txt'
#filename='rf150_2airdiam.txt'
#filename='z2450airdiam.txt'
filename='ef640_3jpair.txt'
filename='ef640_4ajpairdiam.txt'
filename='Nikkor640eairdiam.txt'
filename='Nikkor856eairdiam.txt'
###### incident rays #####
Nray=5
raywidth=0.7
x0=-1e7
h0=-21.5/800.*x0
x_in=900
hray=80
ray_end=501.6
#### evaluation #####
evaluate=False
Neval=50
weval=2.0
#### element location ####
location=0
lenslw=1.
def getpsf(raylist,xfocus,N=100,width=1.0):
  Nray=len(raylist)
  yarr=np.zeros(Nray)
  x0=xfocus
  for i in range(Nray):
    yarr[i]=-raylist[i].A/raylist[i].B*x0-raylist[i].C/raylist[i].B

  ymax=max(yarr)
  ymin=min(yarr)
  wid0=ymax-ymin
  pix=(ymax-ymin)/N
  xradf=np.linspace(ymin,ymax,N+1)
  xradc=(xradf[1:]+xradf[:-1])/2 
  count=np.zeros(N)
  for i in range(Nray):
    j=int(np.floor((yarr[i]-ymin)/pix))
    if j>=0 and j<N:
      count[j]+=1
  idxc=np.argmax(count)
  yc = yarr[idxc]
  ymax = width*wid0/2+yc
  ymin =-width*wid0/2+yc
  #### repeat for new grids
  pix=(ymax-ymin)/N
  xradf=np.linspace(ymin,ymax,N+1)
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
      lens[i_lens]['xc']=xc
      lens[i_lens]['thick']=digtype(surf[i]['d'])

      if surf[i]['type']=='EVENASPH' or surf[i]['type']=='STANDARD':
        lens[i_lens]['parm_L']=digtype(surf[i]['parm'])
        lens[i_lens]['parm_R']=digtype(surf[i+1]['parm'])
        lens[i_lens]['conic_L']=digtype(surf[i]['conic'])
        lens[i_lens]['conic_R']=digtype(surf[i+1]['conic'])
      i_lens+=1
  filtered_lens = [d for d in lens if d]
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
#lens_params=readzmx_asph("mobile-iPhone_USP20170299845.zmx")
#lens_params=readzmx_asph("DoubleGauss_Wakiyama_USP4448497.zmx")
print('lens_params',lens_params)
####### edit some lens #######
#lens_params[4]['n_R']=1.514
#lens_params[5]['n_L']=1.514
for i in range(0*len(lens_params)):
  print(i,lens_params[i]['n_L'],lens_params[i]['n_in'],lens_params[i]['n_R'],'curv',lens_params[i]['curv_L'],lens_params[i]['curv_R'],'pos',lens_params[i]['xc'],lens_params[i]['thick'])
#lens_params[-2]['parm_L']= [0.0,-5.93134E-05,-2.83344E-06,1.13683E-7,-2.41213E-09,2.81858E-11,-1.74792E-13,4.56347E-16]
print('last lens',lens_params[-2])
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
ax1.set_xlim(-25,lens_params[-1]['xc']*1.4)
ax1.set_ylim(-lens_params[0]['diam_L']/1.5,lens_params[0]['diam_L']/1.5)
#ax1.set_xlim(39.3,39.34)
#ax1.set_ylim(5.85,5.88)
#Nray=10
Ray1=[ray2d(0,0,0,0,0,color='C1') for i in range(Nray)]
Ray2=[ray2d(0,0,0,0,0,color='C2') for i in range(Nray)]
Ray3=[ray2d(0,0,0,0,0,color='C3') for i in range(Nray)]
Ray1=gen_optics().ray_in(x0,-h0,x_in,hray,Ray1)
Ray2=gen_optics().ray_in(x0,0.*h0,x_in,hray,Ray2)
Ray3=gen_optics().ray_in(x0,-0.5*h0,x_in,hray,Ray3)
#print('lenRay',len(Ray))
#Ray[:Nray]=gen_optics().ray_in(x0,-h0,x_in,hray,Ray[:Nray])
#Ray[Nray:2*Nray]=gen_optics().ray_in(x0,0*h0,x_in,hray,Ray[Nray:2*Nray])
#Ray[2*Nray:]=gen_optics().ray_in(x0,-0.5*h0,x_in,hray,Ray[2*Nray:])
#Ray=gen_optics().ray_in(x0,h0,1,1,Ray)


lens.assemble(ax1,lw=lenslw,lenscolor='k')
[rayout,rayout1]=lens.raytrace(ax1,Ray1,raywidth=raywidth,rayend=ray_end)
[rayout,rayout1]=lens.raytrace(ax1,Ray3,raywidth=raywidth,rayend=ray_end)
[rayout,rayout1]=lens.raytrace(ax1,Ray2,raywidth=raywidth,rayend=ray_end)

#print('rayout',rayout[0].xs,rayout[0].xe)
#[xfocus,ymax,ymin]=lens.findfocus(rayout,step=2,x0=150,x1=175.)
ydist=np.zeros(Nray)

xfocus= 164.4827880332731
xfocus=164.575
xfocus=125.00
xfocus=75.8015+11.438
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
  ax2=plt.subplot(223)
  #gen_optics().plotray(ax2,100,rayout[0])
  #ax2.set_xlim(-25,200)
  #ax2.set_ylim(-60,60)
  #ax2.set_aspect('equal')
  n=Neval
  [x,count]=getpsf(rayout,xfocus,N=n,width=weval)
  count=count/np.sum(count)
  ft_c = np.abs(np.fft.fft(count))
  d=x[1]-x[0]
  freq = np.fft.fftfreq(n, d)
  half_n = n//2
  ft_c = ft_c[:half_n]
  freq = np.abs(freq[:half_n])  # only non-negative frequencies

  ax2.plot(freq,ft_c)

  ax3=plt.subplot(224)
  ax3.plot(x,count)
#print(x,count)
plt.show()
