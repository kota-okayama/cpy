#!/bin/bash

# OneAPI MKL
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor \
| tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
| tee /etc/apt/sources.list.d/oneAPI.list
apt update
apt install -y intel-oneapi-mkl-devel-2025.0
