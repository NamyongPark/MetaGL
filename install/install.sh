#!/usr/bin/env bash

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/ || exit

CONDA_ENV=MetaGL
if conda info --envs | grep -q "${CONDA_ENV} "; then
  echo "\"${CONDA_ENV}\" conda env exists.";
else
  conda create -y --name "${CONDA_ENV}" python=3.7
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}"/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# install pytorch-1.10.1
if [[ "${OSTYPE}" == "darwin"* ]]; then  # Mac OS
  conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch  # pytorch for Mac
else
  conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch  # pytorch for Linux and Windows
fi

# install dgl-0.8.2post1
conda install -y -c dglteam dgl=0.8.2post1

# install other packages via pip
pip install -r requirements.txt
