import numpy as np
import pandas as pd
import proplot as pplt
import os
from scipy.stats import gaussian_kde
import sys

'''Create Fig. C1, showing how each parameter
affects the bias when varied individually.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig, axs = pplt.subplots(figsize=(10,12),nrows=4,ncols=1,sharey=True, sharex=False)
fig.format(fontsize=16, abc=True, abcloc='ul')
axs.format(ylim=(-18,8),)

df = pd.read_pickle('Data/decadal_synthetics/sigma_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'sigma': sigma}
plotting_df = pd.DataFrame(data=d)
h1 = axs[0].scatter(np.arange(0.02,0.22,step=0.02), plotting_df.groupby('sigma').mean()['time'], label='Decadal \nResolution')
axs[0].format(ylabel='Bias / Years',xlabel='Decadal Resolution Noise / Signal')
axs[0].dualx(lambda x: x*2, label='Annual Resolution Noise / Signal', fontsize = 16, xticks=np.arange(0.04,0.44,0.04))
fit = -253.902084*np.arange(0.02,0.22,step=0.02)**2
h2 = axs[0].plot(np.arange(0.02,0.22,step=0.02), fit, color='black', label='Decadal \nResolution \nFit')

df = pd.read_pickle('Data/annual_synthetics/sigma_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'sigma': sigma}
plotting_df = pd.DataFrame(data=d)
h3 = axs[0].scatter(np.arange(0.02,0.22,step=0.02), plotting_df.groupby('sigma').mean()['time'], marker='s', label='Annual \nResolution')
fit = -42.276479*(2*sigma[:10])**2
h4 = axs[0].plot(sigma[:10], fit, color='black', linestyle='--', label='Annual \nResolution \nFit')


df = pd.read_pickle('Data/decadal_synthetics/GS_slope_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'GS_slope': GS_slope}
plotting_df = pd.DataFrame(data=d)
axs[1].scatter(1000*GS_slope[:10], plotting_df.groupby('GS_slope').mean()['time'], label=None)
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')

fit = -9.507631e+00 + GS_slope[:10] * -1.679905e+03 + GS_slope[:10]**2 * 1.603811e+06
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/annual_synthetics/GS_slope_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'GS_slope': GS_slope}
plotting_df = pd.DataFrame(data=d)
axs[1].scatter(1000*GS_slope[:10], plotting_df.groupby('GS_slope').mean()['time'], marker='s', label=None)
fit = -5.527123 + GS_slope[:10] * -1187.487678
axs[1].plot(1000*GS_slope[:10],fit,color='black', linestyle='--')


df = pd.read_pickle('Data/decadal_synthetics/GIS_slope_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'GIS_slope': GIS_slope}
plotting_df = pd.DataFrame(data=d)
axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(plotting_df.groupby('GIS_slope').mean()['time']), label=None)
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')
fit = -11.103836 + GIS_slope[:10] * -2627.846762
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/annual_synthetics/GIS_slope_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'GIS_slope': GIS_slope}
plotting_df = pd.DataFrame(data=d)
axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(plotting_df.groupby('GIS_slope').mean()['time']), marker='s', label=None)
fit = -7.149843 + GIS_slope[:10]**2 * 754401.882257
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black', linestyle='--')


df = pd.read_pickle('Data/decadal_synthetics/duration_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<500,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)
axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'], label=None)
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')
fit = -12.870503 -0.027344 * duration[:10] +  0.002289 * duration[:10]**2
axs[3].plot(duration[:10], fit, color='black')

df = pd.read_pickle('Data/annual_synthetics/duration_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']
d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)
axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'], marker='s', label=None)
fit = -16.924880 + 0.194995 * duration[:10]
axs[3].plot(duration[:10], fit, color='black', linestyle ='--')

fig.legend((h1,h2,h3,h4), location='right',ncols=1,fontsize=16)
fig.save('Supplementary_Figures/Figure_C1')


'''Create Fig. C2, which shows the bias when
using uniform priors instead of the standard priors.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig, axs = pplt.subplots(figsize=(10,12),nrows=4,ncols=1,sharey=True, sharex=False)
fig.format(fontsize=16)
axs.format(ylim=(-32,3),)

df = pd.read_pickle('Data/decadal_synthetics/sigma_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'sigma': sigma}
sigma_df = pd.DataFrame(data=d)

axs[0].scatter(np.arange(0.02,0.22,step=0.02), sigma_df.groupby('sigma').mean()['time'])
axs[0].format(ylabel='Bias / Years',xlabel='Noise / Signal')

fit = -605.036975*np.arange(0.02,0.22,step=0.02)**2
axs[0].plot(np.arange(0.02,0.22,step=0.02), fit, color='black')

df = pd.read_pickle('Data/decadal_synthetics/GS_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'GS_slope': GS_slope}
GS_df = pd.DataFrame(data=d)

axs[1].scatter(1000*GS_slope[:10], GS_df.groupby('GS_slope').mean()['time'])
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')

fit = -2.148993e+01 + GS_slope[:10] * -6.693412e+03 + GS_slope[:10]**2 * 1.602649e+06
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/decadal_synthetics/GIS_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'GIS_slope': GIS_slope}
GIS_df = pd.DataFrame(data=d)

axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(GIS_df.groupby('GIS_slope').mean()['time']))
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')

fit = -25.565183 + GIS_slope[:10] * -5695.166951
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/decadal_synthetics/dur_alt_prior_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<500,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
time = df['time']

d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)

axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'])
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')

fit = -7.352681 -0.667823 * duration[:10] +  0.005423 * duration[:10]**2

axs[3].plot(duration[:10], fit, color='black')
fig.save('Supplementary_Figures/Figure_C2')


'''Create Fig C3, showing the bias when using the
original Erhardt et al. implementation.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig, axs = pplt.subplots(figsize=(10,12),nrows=4,ncols=1,sharey=True, sharex=False)
fig.format(fontsize=16)
axs.format(ylim=(-28,12),)

df = pd.read_pickle('Data/decadal_synthetics/original_method/sigma_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

d = {'time': time-400, 'sigma': sigma}
sigma_df = pd.DataFrame(data=d)

axs[0].scatter(np.arange(0.02,0.22,step=0.02), sigma_df.groupby('sigma').mean()['time'])
axs[0].format(ylabel='Bias / Years',xlabel='Noise / Signal')
 
fit = 0.678183  -131.271063*np.arange(0.02,0.22,step=0.02)**2
axs[0].plot(np.arange(0.02,0.22,step=0.02), fit, color='black')

df = pd.read_pickle('Data/decadal_synthetics/original_method/GS_slope_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

d = {'time': time-400, 'GS_slope': GS_slope}
GS_df = pd.DataFrame(data=d)

axs[1].scatter(1000*GS_slope[:10], GS_df.groupby('GS_slope').mean()['time'])
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')
 
fit = -6.288424 + GS_slope[:10] * -9620.515442
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/decadal_synthetics/original_method/GIS_slope_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

d = {'time': time-400, 'GIS_slope': GIS_slope}
GIS_df = pd.DataFrame(data=d)

axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(GIS_df.groupby('GIS_slope').mean()['time']))
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')

fit = -2.353384 + GIS_slope[:10] * 2032.128637
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/decadal_synthetics/original_method/duration_pickle')
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)

axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'])
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')
fit = -3.316512 -0.160828 * duration[:10] +  0.003129 * duration[:10]**2

axs[3].plot(duration[:10], fit, color='black')

fig.save('Supplementary_Figures/Figure_C3')


'''Create Fig. C4, showing the bias when using a 
simple least-squares fit.'''

sigma = np.empty(1000)
for i in range(1000):
    sigma[i] = 0.02 + 0.02 * (i%10)

GS_slope = np.empty(1000)
for i in range(1000):
    GS_slope[i] = -1.8e-3+4e-4*(i%10)

GIS_slope = np.empty(1000)
for i in range(1000):
    GIS_slope[i] = -2.7e-3+3e-4*(i%10)

duration = np.empty(1000)
for i in range(1000):
    duration[i] = 10.0 + 10.0 * (i%10)

fig, axs = pplt.subplots(figsize=(10,12),nrows=4,ncols=1,sharey=True, sharex=False)
fig.format(fontsize=16, abc=True, abcloc='ul')
axs.format(ylim=(-16,16),)

df = pd.read_pickle('Data/decadal_synthetics/sigma_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'sigma': sigma}
sigma_df = pd.DataFrame(data=d)
axs[0].scatter(np.arange(0.02,0.22,step=0.02), sigma_df.groupby('sigma').mean()['time'], label='Decadal \nResolution')
axs[0].format(ylabel='Bias / Years',xlabel='Noise / Signal')

df = pd.read_pickle('Data/decadal_synthetics/GS_slope_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'GS_slope': GS_slope}
GS_df = pd.DataFrame(data=d)
axs[1].scatter(1000*GS_slope[:10], GS_df.groupby('GS_slope').mean()['time'], label=None)
axs[1].format(ylabel='Bias / Years',xlabel='Greenland Stadial Slope / $Kiloyears^{-1}$')
fit = GS_slope[:10] * -1.540120e+04 + GS_slope[:10]**3 * 3.028714e+09
axs[1].plot(1000*GS_slope[:10],fit,color='black')

df = pd.read_pickle('Data/decadal_synthetics/GIS_slope_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'GIS_slope': GIS_slope}
GIS_df = pd.DataFrame(data=d)
axs[2].scatter(1000*np.flip(np.abs(GIS_slope[:10])), np.flip(GIS_df.groupby('GIS_slope').mean()['time']), label=None)
axs[2].format(ylabel='Bias / Years',xlabel='Absolute Greenland Interstadial Slope / $Kiloyears^{-1}$')
fit = -1.652046e+00 + GIS_slope[:10]**2 * 1.588440e+06
axs[2].plot(1000*np.flip(np.abs(GIS_slope[:10])),np.flip(fit),color='black')

df = pd.read_pickle('Data/decadal_synthetics/duration_pickle')
df = df.where(df['ls_time']>300,other=np.nan)
df = df.where(df['ls_time']<450,other=np.nan)
time = df['ls_time']
d = {'time': time-400, 'duration': duration}
plotting_df = pd.DataFrame(data=d)
axs[3].scatter(duration[:10], plotting_df.groupby('duration').mean()['time'], label=None)
axs[3].format(ylabel='Bias / Years',xlabel='Transition Duration / Years')

fig.save('Supplementary_Figures/Figure_C4')


''''Create Fig. C5 showing how we down-sample annual resolution
data to decadal resolution'''

low_tau = np.load('Data/low_tau_99.npy')
dec_low_tau = np.load('Data/dec_low_tau_99.npy')
high_both = np.load('Data/high_both_99.npy')
dec_high_both = np.load('Data/dec_high_both_99.npy')
fig, axs = pplt.subplots(nrows=2, ncols=1, figsize=(10,8))
fig.format(fontsize=16)
axs[0].plot(range(800),low_tau)
axs[0].plot(np.arange(800,step=10),dec_low_tau,lw=3)
axs[0].format(title='Whiter Noise', xlabel='Time / Years', fontsize=16)
axs[1].plot(range(800),high_both)
axs[1].plot(np.arange(800,step=10),dec_high_both,lw=3)
axs[1].format(title='Redder Noise', xlabel='Time / Years', fontsize=16)
fig.save('Supplementary_Figures/Figure_C5')


'''Create Fig. C6, which shows a time series
for precipitation in the CCSM4 model. This makes it
clear that there is higher noise during stadials.'''

temp=np.load("Data/CCSM4/PRECIP/200.npy")
fig = pplt.figure(figsize = (8,5),fontsize=16)
ax = fig.subplot(xlabel='Model Year', ylabel='Precipitation / mm $s{^-1}$',fontsize=16)
ax.plot(np.arange(10*len(temp),step=10),temp,label='Precipitation')

fig.save('Supplementary_Figures/Figure_C6')


'''Create Fig. C7, showing the bias due to
separate stadial and interstadial noise regimes.'''

fig, axs = pplt.subplots(figsize=(10,8),nrows=2,ncols=1,sharey=True, sharex=False)
fig.format(fontsize=16)
axs.format(ylim=(-14,-1))

df = pd.read_pickle('Data/decadal_synthetics/systematic_pickle')
df = df.where(df['tau']>1,other=np.nan)
df = df.where((df['sigma']/np.abs(df['jump']))<0.5,other=np.nan)
df = df.where(df['time']>300,other=np.nan)
df = df.where(df['time']<450,other=np.nan)
time = df['time']

sigma1 = np.empty(10000)
sigma2 = np.empty(10000)

for i in range(10000):
    sigma1[i] = 0.03+0.03*int((i%100)/10)
    sigma2[i] = 0.03+0.03*int(i/1000)


d = {'time': time-400, 'sigma1': sigma1, 'sigma2': sigma2}
plotting_df = pd.DataFrame(data=d)

axs[0].scatter(np.arange(0.03,0.30,0.03), plotting_df.groupby('sigma1').mean()['time'])
axs[0].format(ylabel='Bias / Years',xlabel='Stadial Noise / Signal')
axs[1].scatter(np.arange(0.03,0.30,0.03), plotting_df.groupby('sigma2').mean()['time'])
axs[1].format(ylabel='Bias Years',xlabel='Interstadial Noise / Signal')


sigma1_fit =  4.312053 -64.820610 * np.arange(0.03,0.30,0.03) -35.727530 * np.mean(np.arange(0.03,0.30,0.03)) \
            + 182.806446 * np.arange(0.03,0.30,0.03) * np.mean(np.arange(0.03,0.30,0.03))

sigma2_fit =  4.312053 -64.820610 * np.mean(np.arange(0.03,0.30,0.03)) -35.727530 * np.arange(0.03,0.30,0.03) \
            + 182.806446 * np.mean(np.arange(0.03,0.30,0.03)) * np.arange(0.03,0.30,0.03)


axs[0].plot(np.arange(0.03,0.30,0.03), sigma1_fit, color = 'black')
axs[1].plot(np.arange(0.03,0.30,0.03), sigma2_fit, color = 'black')
fig.save('Supplementary_Figures/Figure_C7')