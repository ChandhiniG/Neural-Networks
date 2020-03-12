# from your terminal: ssh NAME@dsmlp-login.ucsd.edu
# in your ssh session: 
#     git clone https://github.com/duckietown/gym-duckietown.git
#     cd gym-duckietown
#     pip3 install -e .
#     vi gym_duckietown/graphics.py 
#         COMMENT OUT LINE 161 and 193
#     cd path/to/gym-duckietown
#     export PYTHONPATH="${PYTHONPATH}:`pwd`"

# Start your pod
#      ./launch_ece253_final.sh -b


#!/bin/sh

set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_CUDA_VERSION=${K8S_CUDA_VERSION:-"9"}
if [ "$K8S_CUDA_VERSION" == "9" ]; then
	K8S_DOCKER_IMAGE=${K8S_PY3TORCH_DOCKER_IMAGE:-"dukelin95/ucsd_rl_pytorch:v2"}
else
	K8S_DOCKER_IMAGE=${K8S_PY3TORCH_DOCKER_IMAGE:-"dukelin95/ucsd_rl_pytorch:v2"}
fi

K8S_ENTRYPOINT="/run_jupyter.sh"

K8S_NUM_GPU=1  # max of 2 (contact ETS to raise limit)
K8S_NUM_CPU=4  # max of 8 ("")
K8S_GB_MEM=16  # max of 64 ("")

# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=YES

# Sets up proxy URL for Jupyter notebook inside
PROXY_ENABLED=YES
PROXY_PORT=8888

exec /software/common64/dsmlp/bin/launch.sh "$@"
