#!/bin/bash

# Preprocessing
source /opt/intel/oneapi/setvars.sh --force

# Execute
exec "$@"

# Postprocessing
