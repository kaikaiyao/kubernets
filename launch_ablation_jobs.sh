#!/bin/bash
set -xe

USER="s2470447-infk8s"
INFK8S_QUEUE_NAME="informatics-user-queue"
STORAGE="250Gi"
DELTA_VALUES=("0.2")

for i in "${!DELTA_VALUES[@]}"; do
    MAX_DELTA=${DELTA_VALUES[$i]}
    INDEX=$i

    # Create temporary files for PVC and Job
    PVC_TEMP_FILE=$(mktemp)
    JOB_TEMP_FILE=$(mktemp)

    # Substitute variables in PVC template
    sed "s/\$USER/$USER/g; s/\$INDEX/$INDEX/g; s/\$STORAGE/$STORAGE/g" pvc-ablation.yml > "$PVC_TEMP_FILE"

    # Substitute variables in Job template
    sed "s/\$USER/$USER/g; s/\$INFK8S_QUEUE_NAME/$INFK8S_QUEUE_NAME/g; s/\$INDEX/$INDEX/g; s/\$MAX_DELTA/$MAX_DELTA/g" job-ablation.yaml > "$JOB_TEMP_FILE"

    # Create PVC
    kubectl create -f "$PVC_TEMP_FILE"

    # Create Job
    kubectl create -f "$JOB_TEMP_FILE"

    # Clean up temporary files
    rm -f "$PVC_TEMP_FILE" "$JOB_TEMP_FILE"
done