Learning expressive control for deep audio synthesis on Eurorack
================================================================

This project aims to implement on constrained hardware platforms 
a light-weight deep generative model generating audio in real-time 
and capable of expressive and understandable control to promote 
creative musical applications.

## Install :

- Clone the github repository :
```bash
$ git clone https://github.com/nabsara/deep-eurorack-control.git
```
- Create a virtual environment with Python 3.8
- Activate the environment and install the dependencies with :
```bash
(myenv)$ pip install -r requirements.txt
```

TODO: add .env configuration


## Project Structure :

```bash 
deep-eurorack-control
├── data    # directory to store the data in local /!\ DO NOT COMMIT /!\
├── docs    # to build Sphinx documentation based on modules docstrings
├── models  # directory to store the models checkpoints in local /!\ DO NOT COMMIT /!\
├── notebooks   # jupyter notebooks for data exploration and models analysis
├── README.md
├── requirements.txt   # python project dependencies with versions
├── scripts   # scripts to executes pipelines (data preparation, training, evaluation) 
├── setup.py
├── src
│   └── deep_eurorack_control  # main package
│       ├── config.py      # global settings based on environment variables
│       ├── datasets       # data preprocessing and dataloader functions
│       │   └── __init__.py
│       ├── helpers        # global utility functions
│       │   └── __init__.py
│       ├── __init__.py
│       ├── models         # models architecture defined as class objects
│       │   └── __init__.py
│       └── pipelines      # pipelines for data preparation, training and evaluation for a given model
│           └── __init__.py
└── tests                        # tests package with unit tests
    ├── conftest.py
    └── __init__.py

```

## Prepare Nsynth dataset files corresponding to the instrument class

```bash
(myenv)$  python -m deep_eurorack_control write-nsynth-json --nsynth_path /CHANGE-ME/nsynth-test --instrument_class string --output_filename nsynth_string_test.json
```

## Run on server

First create `data` and `models` directories

1. Go to deep-eurorack-control directory and open a screen session
```bash
$ screen -S "name"  # to create a new screen

$ screen -r "name"  # to rattach if screen already exists
```
NB : To exit a screen but not delete it do : ctrl + A and then ctrl + D (if you do just ctrl + D)

2. activate python virtual environment. If you use miniconda just do:
```bash
$ source ../miniconda/bin/activate  # change absolute path if needed
```

3. export environment variables:
```bash
$ export DATA_DIR=${PWD}/data
$ export MODELS_DIR=${PWD}/models
$ export AUDIO_DIR=${PWD}/data/audio  # CHANGE if needed
```

4. specify the GPU to use:
check the available GPU with `$ nvidia-smi`
```bash
$ export CUDA_VISIBLE_DEVICE=1  # replace the value with the GPU id you want to use
```

5. Launch training: (example)
```bash
$ python -m deep_eurorack_control train-rave --batch_size 8 --n_epochs 4 --display_step 100 --n_epoch_warmup 2
```
to check available train options :
```bash
$ python -m deep_eurorack_control train-rave --help
```