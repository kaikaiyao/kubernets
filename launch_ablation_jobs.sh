#!/bin/bash

STORAGE="250Gi"
DELTA_VALUES=("0.2")

for i in "${!DELTA_VALUES[@]}"; do
  MAX_DELTA=${DELTA_VALUES[$i]}
  INDEX=$i

  # Create PVC
  envsubst '$USER $INDEX $STORAGE' < pvc-ablation.yml | kubectl create -f -

  # Create Job
  envsubst '$USER $INFK8S_QUEUE_NAME $INDEX $MAX_DELTA' < job-ablation.yaml | kubectl create -f -
done