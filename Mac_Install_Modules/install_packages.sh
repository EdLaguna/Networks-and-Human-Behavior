#!/bash/bin

# First we create a virtual environment for the class
python3 -m venv econ46_virtual_env

# Activate the new environment
source econ46_virtual_env/bin/activate

# install modules that will be used for the code in the class 
pip3 install -U pip
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ jupyter
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ numpy
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ pandas
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ matplotlib
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ scipy
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ networkx
pip3 install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/MacPackages/ ipywidgets

# Once everything was installed we deactivate the environment, 
# i.e. we return to the base environment of your computer.

deactivate