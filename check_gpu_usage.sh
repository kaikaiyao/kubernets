#!/bin/bash

# Specify the namespace you have access to
NAMESPACE="informatics"

# Fetch all pod information in JSON format for the specified namespace
pods=$(kubectl get pods -n "$NAMESPACE" -o json)

# Fetch all node information in JSON format to get GPU type
nodes=$(kubectl get nodes -o json)

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
        
        # Get the GPU type from the node
        gpu_type=$(echo "$nodes" | jq -r --arg node "$node_name" '.items[] | select(.metadata.name == $node) | .metadata.labels."nvidia.com/gpu.product"')
        
        # Store the GPU type for the user if not already set
        if [[ -z "${gpu_types[$user]}" ]]; then
            gpu_types["$user"]=$gpu_type
        fi
    fi
done < <(echo "$pods" | jq -c '.items[]')

# Output the GPU usage per user, sorted by GPU usage in descending order
echo -e "USER\t\tGPUs\tGPU Type"
for user in $(printf "%s\n" "${!gpu_usage[@]}" | sort -nr -k2 -t$'\t' -k1,1); do
    echo -e "$user\t\t${gpu_usage[$user]}\t${gpu_types[$user]}"
done