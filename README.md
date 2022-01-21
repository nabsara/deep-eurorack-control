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
