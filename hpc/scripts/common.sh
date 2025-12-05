alias go=$DCFT_PRIVATE/hpc/scripts/super_go.sh
alias goeval=$DCFT_PRIVATE/hpc/scripts/goeval.sh
alias sloweval=$DCFT_PRIVATE/hpc/scripts/sloweval.sh
alias fullsloweval=$DCFT_PRIVATE/hpc/scripts/fullsloweval.sh
alias fulleval=$DCFT_PRIVATE/hpc/scripts/fulleval.sh
alias gotrain=$DCFT_PRIVATE/hpc/scripts/gotrain.sh
alias gomicro=$DCFT_PRIVATE/hpc/scripts/gomicro.sh
alias gosmall=$DCFT_PRIVATE/hpc/scripts/gosmall.sh
alias golarge=$DCFT_PRIVATE/hpc/scripts/golarge.sh
alias gofast=$DCFT_PRIVATE/hpc/scripts/gofast.sh
alias goscale=$DCFT_PRIVATE/hpc/scripts/goscale.sh
# Keep the old go.sh script accessible as legacy_go
alias clearhf=$DCFT_PRIVATE/hpc/scripts/clearhf.sh
alias legacy_go=$DCFT_PRIVATE/hpc/scripts/go.sh
alias sfail=$DCFT_PRIVATE/hpc/scripts/sfailed.sh
alias swin=$DCFT_PRIVATE/hpc/scripts/scompleted.sh
alias soops=$DCFT_PRIVATE/hpc/scripts/scancelled.sh
alias rmlogs=$DCFT_PRIVATE/hpc/scripts/rmlogs.sh
alias status=$DCFT_PRIVATE/hpc/scripts/status.sh
alias scancelall=$DCFT_PRIVATE/hpc/scripts/scancelall.sh
alias upload=$DCFT_PRIVATE/hpc/scripts/upload.sh
alias sinf='sinfo -S+P -o "%18P %8a %20F"'
alias sqme='squeue -u $USER --format="%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r"'
alias sqthem="squeue --format=\"%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r\" -u "
alias sqteam="squeue --format=\"%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r\" -A $(sacctmgr -n show associations user=$USER format=account%30 | awk 'NR==1{print $1}' | xargs)"
alias dcft='cd $DCFT_PRIVATE && $DCFT_PRIVATE_ACTIVATE_ENV'
alias evalchemy='cd $EVALCHEMY && $EVALCHEMY_ACTIVATE_ENV'
# Define the alias to start and track the auto upload process
alias start_auto_upload='nohup $DCFT_PRIVATE/hpc/scripts/auto_upload.sh > $DCFT_PRIVATE/hpc/scripts/auto_upload.log 2>&1 & echo $! > $DCFT_PRIVATE/hpc/scripts/auto_upload.pid && echo "Background process started with PID $(cat $DCFT_PRIVATE/hpc/scripts/auto_upload.pid)"'
alias stop_auto_upload='kill $(cat $DCFT_PRIVATE/hpc/scripts/auto_upload.pid)'
alias status_auto_upload='cat $DCFT_PRIVATE/hpc/scripts/auto_upload.log'
# Load database credentials if present
if [ -f "$DCFT_PRIVATE/database/access.env" ]; then
  source "$DCFT_PRIVATE/database/access.env"
else
  echo "[hpc/common.sh] database/access.env not found; continuing without DB creds" >&2
fi
# for shared files
if [[ "$PRIMARY_GROUP_SET" != "true" ]]; then
  export PRIMARY_GROUP_SET=true
  exec newgrp $DCFT_GROUP
fi
umask 007
