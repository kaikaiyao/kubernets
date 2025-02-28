#!/bin/bash
# This script aggregates GPU usage per user by parsing the output of "kubectl describe pods"

# Declare an associative array to hold GPU usage per user
declare -A gpu_usage
current_user=""

# Process the kubectl describe pods output line-by-line
while IFS= read -r line; do
    # When a new pod starts, clear the current user
    if [[ "$line" =~ ^Name: ]]; then
         current_user=""
    fi

    # Look for the label line that contains the user info (e.g., "eidf/user=s2027538-infk8s")
    if [[ "$line" =~ "eidf/user=" ]]; then
         if [[ "$line" =~ eidf\/user=([^[:space:]]+) ]]; then
             current_user="${BASH_REMATCH[1]}"
         fi
    fi

    # Look for the GPU limit line; assume it is formatted like "nvidia.com/gpu:  2"
    if [[ "$line" =~ "nvidia.com/gpu:" ]]; then
         # Extract the GPU count (assumed to be the second field)
         gpu_count=$(echo "$line" | awk '{print $2}')
         # Only add if we have a user associated with this pod
         if [[ -n "$current_user" ]]; then
              # Add the GPU count to the current user's total
              gpu_usage["$current_user"]=$(( gpu_usage["$current_user"] + gpu_count ))
         fi
    fi
done < <(kubectl describe pods)

# Output the aggregated GPU usage per user
echo "GPU usage per user:"
for user in "${!gpu_usage[@]}"; do
    echo "$user: ${gpu_usage[$user]} GPU(s)"
done
