import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nbformat import from_dict
import numpy as np


from deep_eurorack_control.models.ddsp.ops import get_loudness


def plot_harmonics(harmonics,ax=None,fig=None):
    def add_colorbar(fig, ax, im):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    harms = np.arange(harmonics.shape[1])
    times = np.arange(harmonics.shape[0])
    X, Y = np.meshgrid(times, harms)

    im0 = ax.pcolor(X, Y, harmonics.T, cmap="magma")

    add_colorbar(fig, ax, im0)
    labelRow = "Echantillon"
    labelCol = "Harmoniques"
    ax.set_xlabel(labelRow)
    ax.set_ylabel(labelCol)


def plot_metrics(pitch,loud,signal,harmonics,sr,frame_size):
    
    level = harmonics[:,0]
    harmonics_mag = harmonics[:,1:]/np.sum(harmonics[:,1:])
    loud_out = get_loudness(signal,sr,frame_size,n_fft=1024)
    
    fig = plt.figure(figsize=(18,10))
    spec = fig.add_gridspec(3,2)
    ax_pitch = fig.add_subplot(spec[0, 0])
    ax_loud = fig.add_subplot(spec[1, 0])
    ax_level = fig.add_subplot(spec[2, 0])
    ax_harm = fig.add_subplot(spec[:, 1])
    
    
    times = np.linspace(0,pitch.shape[0]*frame_size/sr,pitch.shape[0])
    ax_pitch.plot(times,pitch,color='k',linestyle='--',label='Target')
    
    ax_loud.plot(times,loud_out,label='Reconstructed')
    ax_loud.plot(times,loud,color='k',linestyle='--',label='Target')
    
    ax_level.plot(times,level,label='Reconstructed')

    ax_loud.set_title('Loudness')
    ax_loud.set_ylabel('Loudness')
    ax_loud.legend()
    

    ax_pitch.set_title('Pitch')
    ax_pitch.set_ylabel('Frequency (Hz)')
    ax_pitch.legend()
    
    ax_level.set_title('Level')
    ax_level.set_ylabel('Level')
    ax_level.set_xlabel('Time (s)')
    ax_level.set_ylim(0,1)
    ax_level.legend()
    

    plot_harmonics(harmonics_mag,ax_harm,fig,times)
    return (fig)
        
