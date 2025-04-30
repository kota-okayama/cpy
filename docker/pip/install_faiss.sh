#!/bin/bash

# FAISS v1.9.0 (for python)
source /opt/intel/oneapi/setvars.sh

# Install gflags
apt update
apt install -y libgflags-dev

# Select compiler
# Cannot build with gcc-13
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# Build and install
git clone --depth 1 --branch v1.10.0 https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build . \
    -DBLA_VENDOR=Intel10_64_dyn \
    -DMKL_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.2
# -j option is for parallel build
make -C build -j64 faiss
make -C build -j64 faiss_avx2
make -C build -j64 swigfaiss
cd build/faiss/python
python -m pip install --break-system-packages .

# Delete unnecessary files
cd ../../../..
rm -rf faiss
