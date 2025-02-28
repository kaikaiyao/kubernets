#!/bin/bash

# Specify the namespace you have access to
NAMESPACE="informatics"

# Fetch all pod information in JSON format for the specified namespace
pods=$(kubectl get pods -n "$NAMESPACE" -o json)

# Initialize an associative array to store GPU usage per user
declare -A gpu_usage
declare -A gpu_types

# Parse the JSON output for pods
while read -r line; do
    # Extract the user label
    user=$(echo "$line" | jq -r '.metadata.labels."eidf/user"')
    
    # Extract the GPU request
    gpu_request=$(echo "$line" | jq -r '.spec.containers[].resources.requests."nvidia.com/gpu"')
    
    # Extract the node name
    node_name=$(echo "$line" | jq -r '.spec.nodeName')
    
    # Check if both user and gpu_request are not null
    if [[ "$user" != "null" && "$gpu_request" != "null" ]]; then
        # Add the GPU request to the user's total GPU usage
        gpu_usage["$user"]=$((${gpu_usage["$user"]:-0} + gpu_request))
        
        # Get the GPU type from the pod's node labels if available
        gpu_type=$(echo "$line" | jq -r '.spec.nodeSelector."nvidia.com/gpu.product"')
        
        # If GPU type is not found in nodeSelector, try to get it from annotations
        if [[ "$gpu_type" == "null" ]]; then
            gpu_type=$(echo "$line" | jq -r '.metadata.annotations."nvidia.com/gpu.product"')
        fi
        
        # If GPU type is still not found, set it to "Unknown"
        if [[ "$gpu_type" == "null" ]]; then
            gpu_type="Unknown"
        fi
        
        # Store the GPU type for the user if not already set
        if [[ -z "${gpu_types[$user]}" ]]; then
            gpu_types["$user"]=$gpu_type
        fi
    fi
done < <(echo "$pods" | jq -c '.items[]')

# Output the GPU usage per user, sorted by GPU usage in descending order
echo -e "USER\t\tGPUs\tGPU Type"
for user in $(for u in "${!gpu_usage[@]}"; do echo -e "$u\t${gpu_usage[$u]}\t${gpu_types[$u]}"; done | sort -nr -k2); do
    echo "$user"
done