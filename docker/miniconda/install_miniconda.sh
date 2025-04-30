# install miniconda for ubuntu 22.04

# miniconda
# Ref: https://docs.anaconda.com/free/miniconda/
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
chmod +x /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda
. /opt/miniconda/bin/activate
conda init bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda create -n saturn python=3.11.8
conda activate saturn
conda env update --file /tmp/environment.yml
