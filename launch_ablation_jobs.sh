#!/bin/bash
set -xe 

export STORAGE="10Gi"  # Export STORAGE
DELTA_VALUES=("2.0" "1.0" "0.5" "0.2" "0.1" "0.05" "0.02" "0.01")

for i in "${!DELTA_VALUES[@]}"; do
  export MAX_DELTA=${DELTA_VALUES[$i]}  # Export MAX_DELTA
  export INDEX=$i  # Export INDEX

  # Create PVC
  envsubst '$USER $INDEX $STORAGE' < pvc-ablation.yml | kubectl create -f -

  # Create Job
  envsubst '$USER $INFK8S_QUEUE_NAME $INFK8S_NFS_SERVER_IP $INDEX $MAX_DELTA' < job-ablation.yaml | kubectl create -f -
done