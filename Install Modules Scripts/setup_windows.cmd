
py -m venv econ46_virtual_env

call econ46_virtual_env\Scripts\activate.bat

py -m pip install --upgrade pip
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ jupyter
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ numpy
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ pandas
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ matplotlib
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ scipy
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/ networkx==2.3
py -m pip install --no-index --find-links https://web.stanford.edu/~jacksonm/econ46/Python/WinPackages/ ipywidgets

ipython kernel install --user --name=econ46_virtual_env

deactivate