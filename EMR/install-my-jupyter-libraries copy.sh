#!/bin/bash

sudo easy_install-3.4 pip
sudo /usr/local/bin/pip3 install scipy scikit-learn pandas boto3 matplotlib plotly awscli
sudo apt-get install nfs-common
# export PATH=~/.local/bin:$PATH