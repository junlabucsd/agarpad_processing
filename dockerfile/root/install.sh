#!/bin/bash

echo "update sources and upgrade distro"
apt-get update
apt dist-upgrade -y

echo "install python version in repositories"
apt-get install -y python python-dev python-pip python-tk

echo "install other packages for imaging and plotting"
apt-get install -y libreadline-dev libsqlite3-dev libbz2-dev libssl-dev
apt-get install -y libblas-dev liblapack-dev libatlas-dev
apt-get install -y libpng-dev libfreetype6-dev tk-dev pkg-config

echo "install additional python packages with pip"
python -m ensurepip
python -m pip install pip --upgrade
python -m pip install wheel

python -m pip install numpy scipy matplotlib Pillow pyYAML termcolor opencv-python tifffile pims_nd2

echo "finished."
