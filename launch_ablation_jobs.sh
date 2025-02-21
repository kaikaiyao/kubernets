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
    sed "s/\$USER/$USER/g; s/\$INDEX/$INDEX/g; s/\$STORAGE/$STORAGE/g" pvc-ablation.yml | kubectl create -f -

    # Create Job
    sed "s/\$USER/$USER/g; s/\$INFK8S_QUEUE_NAME/$INFK8S_QUEUE_NAME/g; s/\$INDEX/$INDEX/g; s/\$MAX_DELTA/$MAX_DELTA/g" job-ablation.yaml | kubectl create -f -
done