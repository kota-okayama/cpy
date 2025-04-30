#!/bin/bash

# fastText v0.9.3 (archived master branch)
git clone --depth 1 https://github.com/facebookresearch/fastText.git
cd fastText
pip install --break-system-packages .
cd ..
rm -r fastText
