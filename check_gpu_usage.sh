#!/bin/bash

# Specify your namespace explicitly
NAMESPACE="informatics"

# Get pods in your namespace and capture errors
output=$(kubectl describe pods -n "$NAMESPACE" 2>&1)
exit_code=$?

# Check for errors
if [ $exit_code -ne 0 ]; then
    echo "Error getting pods:"
    echo "$output" | grep -i "forbidden\|error"
    exit 1
fi

# Process the output
echo "$output" | awk '
BEGIN {
    RS = "";  # Pod blocks separated by blank lines
    FS = "\n";
    total_gpus[""] = 0;  # Initialize array
}

{
    status = "";
    user = "";
    gpus = 0;
    in_limits = 0;

    for (i = 1; i <= NF; i++) {
        line = $i;

        # Check pod status
        if (line ~ /^Status:[[:space:]]+Running/) {
            status = "Running";
        }

        # Extract user label
        if (line ~ /Labels:.*eidf\/user=/) {
            split(line, parts, "eidf/user=");
            split(parts[2], user_parts, /[, ]/);
            user = user_parts[1];
        }

        # Track GPU limits
        if (line ~ /Limits:/) { in_limits = 1 }
        if (line ~ /Requests:/) { in_limits = 0 }
        if (in_limits && line ~ /nvidia.com\/gpu:/) {
            split(line, gpu_parts, /nvidia.com\/gpu:[[:space:]]+/);
            gpus += gpu_parts[2] + 0;
        }
    }

    if (status == "Running" && user != "") {
        total_gpus[user] += gpus;
    }
}

END {
    printf "%-20s %s\n", "USER", "GPUs";
    for (user in total_gpus) {
        if (user != "") {  # Skip empty users
            printf "%-20s %d\n", user, total_gpus[user];
        }
    }
}'