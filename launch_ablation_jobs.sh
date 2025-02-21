#!/bin/bash
set -xe

USER="s2470447-infk8s"
INFK8S_QUEUE_NAME="informatics-user-queue"
STORAGE="250Gi"
DELTA_VALUES=("0.2")

for i in "${!DELTA_VALUES[@]}"; do
  MAX_DELTA=${DELTA_VALUES[$i]}
  INDEX=$i

  # Create PVC
  PVC_YAML=$(envsubst '$USER $INDEX $STORAGE' < pvc-ablation.yml)
  echo "Creating PVC for INDEX=$INDEX"
  echo "$PVC_YAML"
  echo "$PVC_YAML" | kubectl create -f -

  # Create Job
  JOB_YAML=$(envsubst '$USER $INFK8S_QUEUE_NAME $INDEX $MAX_DELTA' < job-ablation.yaml)
  echo "Creating Job for INDEX=$INDEX"
  echo "$JOB_YAML"
  echo "$JOB_YAML" | kubectl create -f -
done