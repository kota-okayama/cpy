#!/bin/bash

# Install MKL Numpy and Scipy
# https://zenn.dev/fate_shelled/scraps/adcf327fd33425
# https://qiita.com/Shunmo17/items/0ec675992bc25da4cdf0

source /opt/intel/oneapi/setvars.sh

apt update
apt install -y autoconf automake libtool  # common
apt install -y meson libatlas-base-dev gfortran pkg-config  # for scipy
pip uninstall --break-system-packages -y numpy scipy

# Install MKL Numpy 2.0.2 (Tensorflow 2.19.0 requires numpy<2.2.0)
# Cannot build: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory (Numpy 2.1.0)
pip --break-system-packages install cython
git clone --depth 1 --recursive --branch v2.1.1 https://github.com/numpy/numpy.git
cd numpy
cat <<EOF > site.cfg
[mkl]
library_dirs = /opt/intel/oneapi/mkl/latest/lib
include_dirs = /opt/intel/oneapi/mkl/latest/include
mkl_libs = mkl_rt
lapack_libs = mkl_rt
EOF
pip install --break-system-packages .
cd ..

# Install MKL Scipy
# https://github.com/scipy/scipy/issues/16200
# https://docs.scipy.org/doc/scipy-1.14.1/building/blas_lapack.html
pip install --break-system-packages build pythran pybind11 meson-python ninja pydevtool rich-click
git clone --depth 1 --recursive --branch v1.15.2 https://github.com/scipy/scipy.git
cd scipy
pip install --break-system-packages . \
    -Csetup-args=-Dblas=mkl-sdl \
    -Csetup-args=-Dlapack=mkl-sdl
python -c "import scipy; scipy.show_config()"
cd ..

# Delete unnecessary files
rm -rf numpy scipy
