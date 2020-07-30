#! /bin/bash

set -e

# set name of enviornment
ENV_NAME=fiteos

# store location of this script
TOOLS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# store Python version
PYTHON_VERSION=3.7.3

# import Anaconda functions into bash script
source ${CONDA_PREFIX}/etc/profile.d/conda.sh

# clean environment
conda activate
conda env remove --yes --name ${ENV_NAME}

# create Anaconda environment
conda create --yes --name ${ENV_NAME} python=${PYTHON_VERSION}
conda activate ${ENV_NAME}

# install packages
conda install -y pandas==1.0.3 numpy==1.18.1 matplotlib==3.1.3 scipy==1.4.1
conda install -y -c conda-forge lmfit==1.0.0 uncertainties==3.1.2

# install pyeos
cd ${TOOLS_DIR}/..
python setup.py install
