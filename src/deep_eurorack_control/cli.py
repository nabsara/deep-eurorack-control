import click
import os
from deep_eurorack_control.config import settings

from deep_eurorack_control.pipelines.ddsp.train_pipeline import DDSP_Pipeline
from deep_eurorack_control.pipelines.ddsp.dataset_process import preprocess_dataset

@click.option(
    "--sr",
    default=16000,
    help="Audio Sample Rate",
)
@click.option(
    "--dataset_dir",
    default=r"C:\Users\NILS\Documents\ATIAM\PAM\Datasets\pitch_conf_test",
    help="Dataset Location",
)
@click.option(
    "--frame_size",
    default=256,
    help="Frame size for audio subdivision",
)
@click.option(
    "--n_harmonics",
    default=11,
    help="Number of harmonics for the harmonic synthesizer",
)
@click.option(
    "--n_bands",
    default=25,
    help="Number of frequency bands for the substractive synthesizer",
)
@click.option(
    "--lr",
    default=0.001,
    help="Learning Rate",
)
@click.option(
    "--batch_size",
    default=4,
    help="Batch Size",
)
@click.option(
    "--n_epochs",
    default=50,
    help="Number of epochs",
)
@click.option(
    "--display_step",
    default=5,
    help="Los and Tensorboard display step",
)
@click.option(
    "--logdir",
    default=r"C:\Users\NILS\Documents\ATIAM\PAM\deep-eurorack-control\data",
    help="Logs and model checkpoints directory",
)
@click.option(
    "--preprocess",
    default=False,
    help="Preprocess the data (bool)",
)
@click.option(
    "--filters",
    default="string_acoustic",
    help="Dataset to process(if preprocess==True)",
)
@click.option(
    "--raw_data_dir",
    default=r"C:\Users\NILS\Documents\ATIAM\PAM\Datasets\nsynth-test\audio",
    help="Raw Dataset location(if preprocess==True)",
)
@click.option(
    "--alpha",
    default=0.95,
    help="Learning rate decay",
)
@click.option(
    "--residual",
    default=True,
    help="Residual latent space on",
)
@click.option(
    "--n_z",
    default=16,
    help="Residual latent space dim",
)
@click.option(
    "--pitch_estim",
    default=True,
    help="Pitch estimation on",
)
def train_ddsp(dataset_dir,sr,frame_size,n_harmonics,n_bands,lr,batch_size,n_epochs,display_step,logdir,preprocess,filters,raw_data_dir,alpha,residual,n_z,pitch_estim):
    if preprocess==True:
        preprocess_dataset(raw_data_dir,dataset_dir,filters,sr,frame_size,nb_files=None)
    pipeline =DDSP_Pipeline(dataset_dir,sr,frame_size,n_harmonics,n_bands,residual,n_z,pitch_estim)   
    pipeline.train(lr,batch_size,n_epochs,display_step,logdir,alpha)
    
    