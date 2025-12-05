#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a,h] ]]; then nv="--nv"; fi

singularity \
  exec $nv \
  --overlay /scratch/$USER/dcft_singularity/dc_agents.ext3:ro \
  /scratch/$USER/conda_envs/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -c "
 source /ext3/env.sh;
 $args 
"
