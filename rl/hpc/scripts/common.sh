HELPER_SCRIPTS_DIR="$OT_AGENT_RL/hpc/scripts/helpers"
alias sfail="$HELPER_SCRIPTS_DIR/sfailed.sh"
alias scompleted="$HELPER_SCRIPTS_DIR/scompleted.sh"
alias scancelled="$HELPER_SCRIPTS_DIR/scancelled.sh"
alias rmlogs="$HELPER_SCRIPTS_DIR/rmlogs.sh"
alias status="$HELPER_SCRIPTS_DIR/status.sh"
alias scancelall="$HELPER_SCRIPTS_DIR/scancelall.sh"
alias sinf='sinfo -S+P -o "%18P %8a %20F"'
alias sqme='squeue -u $USER --format="%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r"'
alias sqthem="squeue --format=\"%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r\" -u "
alias sqteam="squeue --format=\"%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r\" -A $(sacctmgr -n show associations user=$USER format=account%30 | awk 'NR==1{print $1}' | xargs)"

# Define the alias to start and track the auto upload process
alias start_auto_upload='nohup $OT_AGENT_RL/hpc/scripts/auto_upload.sh > $OT_AGENT_RL/hpc/scripts/auto_upload.log 2>&1 & echo $! > $OT_AGENT_RL/hpc/scripts/auto_upload.pid && echo "Background process started with PID $(cat $OT_AGENT_RL/hpc/scripts/auto_upload.pid)"'
alias stop_auto_upload='kill $(cat $OT_AGENT_RL/hpc/scripts/auto_upload.pid)'
alias status_auto_upload='cat $OT_AGENT_RL/hpc/scripts/auto_upload.log'

# for shared files
if [[ "$PRIMARY_GROUP_SET" != "true" ]]; then
  export PRIMARY_GROUP_SET=true
  exec newgrp $OT_AGENT_GROUP
fi
umask 007
