#!/bin/bash
# Install SC2 and add the custom maps

# Clone the source code.
#git clone git@github.com:tjuHaoXiaotian/pymarl3.git
export PYMARL3_CODE_DIR=$(pwd)
old_dir=$(pwd)
# 0.configure basic env, make sure cd to the project dir
conda init
source ~/.bashrc
conda create -n pymarl python=3.9
conda activate pymarl
# 1. Install StarCraftII
echo 'Install StarCraftII...'
cd "$HOME"
export SC2PATH="$HOME/StarCraftII"
echo 'SC2PATH is set to '$SC2PATH
if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

# 2. Install the custom maps

# Copy the maps to the target dir.
echo 'Install SMACV1 and SMACV2 maps...'
MAP_DIR="$SC2PATH/Maps/"
if [ ! -d "$MAP_DIR/SMAC_Maps" ]; then
    echo 'MAP_DIR is set to '$MAP_DIR
    if [ ! -d $MAP_DIR ]; then
            mkdir -p $MAP_DIR
    fi
    cp -r "$PYMARL3_CODE_DIR/src/envs/smac_v2/official/maps/SMAC_Maps" $MAP_DIR
else
    echo 'SMACV1 and SMACV2 maps are already installed.'
fi
echo 'StarCraft II and SMAC maps are installed.'

echo 'export SC2PATH=~/StarCraftII/' >> ~/.bashrc
source ~/.bashrc
conda activate pymarl

echo 'Install PyTorch and Python dependencies...'

pip install torch==2.2.0
pip install sacred scipy gym==0.10.8 matplotlib seaborn \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger

# pip install git+https://github.com/oxwhirl/smac.git
# Do not need install SMAC anymore. We have integrated SMAC-V1 and SMAC-V2 in pymarl3/envs.
pip install "protobuf<3.21"
pip install "pysc2>=3.0.0"
pip install "s2clientprotocol>=4.10.1.75800.0"
pip install "absl-py>=0.1.0"
pip install "sacred"
pip install "numpy==1.26.4"



cd $old_dir
