module load cray-python
module swap PrgEnv-intel PrgEnv-gnu
module swap craype-mic-knl craype-haswell
export CC='cc -dynamic -march=x86-64'
export MPICC='cc -dynamic -march=x86-64'
export CXX='CC -dynamic -march=x86-64'
export MPICXX='CC -dynamic -march=x86-64'

pip install --upgrade pip wheel setuptools
pip install MDAnalysis
pip install nglview
pip install seaborn
pip install notebook
pip install jupyter_contrib_nbextensions
