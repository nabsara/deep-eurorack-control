"""* run:
pip install .
in this directory to create the python module
then run test.py"""
from setuptools import Extension, setup

# The header file is there in order to separate the actual c implementations
# from the wrapper
module = Extension("tediumControl",
                  sources=[
                    'adc.c',
                    # 'switches.c',
                    'wrapper.c'
                  ])

setup(name='tediumControl',
     version='1.0',
     description="""Python wrapper for custom C extension,
to control the terminal tedium's adc and switch buttons""",
     ext_modules=[module]
)
