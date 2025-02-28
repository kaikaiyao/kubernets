#!/bin/bash

# Specify the namespace you have access to
NAMESPACE="informatics"

# Fetch all pod information in JSON format for the specified namespace
pods=$(kubectl get pods -n "$NAMESPACE" -o json)

# Initialize an associative array to store GPU usage per user
declare -A gpu_usage

# Parse the JSON output
while read -r line; do
    # Extract the user label
    user=$(echo "$line" | jq -r '.metadata.labels."eidf/user"')
    
    # Extract the GPU request
    gpu_request=$(echo "$line" | jq -r '.spec.containers[].resources.requests."nvidia\.com/gpu"')
    
    # Check if both user and gpu_request are not null
    if [[ "$user" != "null" && "$gpu_request" != "null" ]]; then
        # Add the GPU request to the user's total GPU usage
        gpu_usage["$user"]=$((${gpu_usage["$user"]:-0} + gpu_request))
    fi
done < <(echo "$pods" | jq -c '.items[]')

# Output the GPU usage per user
echo "USER                 GPUs"
for user in "${!gpu_usage[@]}"; do
    echo "$user                 ${gpu_usage[$user]}"
done