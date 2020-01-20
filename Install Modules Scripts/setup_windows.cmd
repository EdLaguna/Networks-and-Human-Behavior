
py -m venv econ46_virtual_env

py -m pip install --upgrade pip
py -m pip install jupyter

call econ46_virtual_env\Scripts\activate.bat

py -m pip install --upgrade pip
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ jupyter
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ numpy
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ pandas
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ matplotlib
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ scipy
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ ipywidgets
py -m pip install networkx==2.3

ipython kernel install --user --name=econ46_virtual_env

deactivate