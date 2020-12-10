module load cray-python
module swap PrgEnv-intel PrgEnv-gnu
export CC=cc
export CXX=CC

pip install --upgrade pip wheel setuptools
pip install MDAnalysis
pip install nglview
pip install seaborn
pip install notebook
pip install jupyter_contrib_nbextensions
