import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

from deep_eurorack_control.models.ddsp.ops import get_loudness


def plot_harmonics(harmonics,ax=None,fig=None,times=None):
    def add_colorbar(fig, ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    harms = np.arange(harmonics.shape[1])
    if times is None:
        times = np.arange(harmonics.shape[0])


    X, Y = np.meshgrid(times, harms)

    im0 = ax.pcolor(X, Y, harmonics.T, cmap="magma",shading='auto')

    add_colorbar(fig, ax, im0)
    labelRow = "Time (s)"
    labelCol = "Harmonics"
    # ax.set_title('Harmonic distribution')
    # ax.set_xlabel(labelRow)
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel(labelCol)
    
def plot_filters(filters,sr,ax=None,fig=None,times=None):
    def add_colorbar(fig, ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    freqs = np.linspace(0,sr/2,filters.shape[1])
    
    if times is None:
        times = np.arange(filters.shape[0])


    X, Y = np.meshgrid(times, freqs)

    im0 = ax.pcolor(X, Y, filters.T, cmap="magma",shading='auto')

    add_colorbar(fig, ax, im0)
    labelRow = "Time (s)"
    labelCol = "Filters Frequency (Hz)"
    # ax.set_title('Harmonic distribution')
    ax.set_xlabel(labelRow)
    ax.set_ylabel(labelCol)
 
def plot_stft(stft,sr,ax=None,fig=None,times=None,label=None):
    def add_colorbar(fig, ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    freqs = np.linspace(0,sr/2,stft.shape[0])
    
    if times is None:
        times = np.arange(stft.shape[1])


    X, Y = np.meshgrid(times, freqs)

    im0 = ax.pcolor(X, Y, 20*np.log10(np.abs(stft)+1e-6), cmap="magma",shading='auto')

    add_colorbar(fig, ax, im0)
    labelRow = "Time (s)"
    # labelCol = "Frequency (Hz)"
    ax.set_title(label)
    # ax.set_xlabel(labelRow)
    # ax.set_ylabel(labelCol)
    
       
    
    
def plot_metrics(pitch,signal_in,signal_out,harmonics,filters,sr,frame_size):

    level = harmonics[:,0]
    harmonics_mag = harmonics[:,1:]/np.sum(harmonics[:,1:],axis=-1,keepdims=True)
    loud_out = get_loudness(signal_out,sr,frame_size,n_fft=1024)
    loud = get_loudness(signal_in,sr,frame_size,n_fft=1024)
    
    
    stft_in =  torch.abs(torch.stft(torch.tensor(signal_in),n_fft = 1024,return_complex=True,normalized=False,window=torch.hann_window(1024)) ).numpy() 
    stft_out = torch.abs(torch.stft(torch.tensor(signal_out),n_fft = 1024,return_complex=True,normalized=False,window=torch.hann_window(1024))).numpy()
    
    
    fig = plt.figure(figsize=(20,6))
    spec = fig.add_gridspec(4,3)
    
    ax_pitch = fig.add_subplot(spec[0, 0])
    ax_loud = fig.add_subplot(spec[1, 0])
    ax_level = fig.add_subplot(spec[2, 0])
    ax_wave = fig.add_subplot(spec[3, 0])

    ax_harm = fig.add_subplot(spec[:2, 1])
    ax_filters = fig.add_subplot(spec[2:, 1])
    
    ax_stft_in = fig.add_subplot(spec[:2, 2])
    ax_stft_out = fig.add_subplot(spec[2:, 2])   
    
    
    times = np.linspace(0,pitch.shape[0]*frame_size/sr,pitch.shape[0])
    ax_pitch.plot(times,pitch,color='k',linewidth = 0.5,label='Target')
    
    ax_loud.plot(times,loud_out,label='Reconstructed')
    ax_loud.plot(times,loud,color='k',linestyle='--',label='Target')
    
    ax_level.plot(times,level,label='Reconstructed')
    
    ax_wave.plot(signal_in[:1024],alpha=0.8,color='k',linewidth=0.6)
    ax_wave.plot(signal_out[:1024],linewidth=0.6)
    
    ax_pitch.set_ylabel('Pitch (Hz)')
    # ax_pitch.legend()
    ax_pitch.get_xaxis().set_visible(False)
    ax_loud.set_ylabel('Loudness')
    # ax_loud.legend()
    ax_loud.get_xaxis().set_visible(False)

    ax_level.set_ylabel('Level')
    ax_level.set_ylim(0,1.1)
    # ax_level.legend()
    ax_level.get_xaxis().set_visible(False)

    
    ax_wave.set_xlabel('Sample')
    ax_wave.set_ylabel('Waveform')
    
    fig.align_ylabels([ax_loud,ax_level,ax_pitch,ax_wave])
    fig.align_ylabels([ax_harm,ax_filters])
    # fig.tight_layout()

    plot_harmonics(harmonics_mag,ax_harm,fig,times)
    plot_filters(filters,sr,ax_filters,fig,times)
    plot_stft(stft_in,sr,ax_stft_in,fig,None,'STFT IN')
    plot_stft(stft_out,sr,ax_stft_out,fig,None,'STFT OUT')

    ax_stft_in.get_xaxis().set_visible(False)
    ax_stft_out.get_xaxis().set_visible(False)
    
    return (fig)
        
