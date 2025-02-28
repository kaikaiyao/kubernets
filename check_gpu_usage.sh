#!/bin/bash

kubectl describe pods | awk '
BEGIN {
    RS = "";  # Split input into pod blocks separated by blank lines
    FS = "\n";
}

{
    status = "";
    user = "";
    gpus = 0;
    in_limits = 0;

    for (i = 1; i <= NF; i++) {
        line = $i;

        # Check if the pod is Running
        if (line ~ /^Status:[[:space:]]+Running/) {
            status = "Running";
        }

        # Extract user from Labels
        if (line ~ /Labels:/) {
            split(line, lbl_parts, "eidf/user=");
            if (length(lbl_parts) >= 2) {
                split(lbl_parts[2], user_part, /[, ]/);
                user = user_part[1];
            }
        }

        # Track if we are in the Limits section
        if (line ~ /Limits:/) {
            in_limits = 1;
        }
        if (line ~ /Requests:/) {
            in_limits = 0;
        }

        # Extract GPU count from Limits
        if (in_limits && line ~ /nvidia.com\/gpu:/) {
            split(line, gpu_parts, /nvidia.com\/gpu:[[:space:]]+/);
            if (length(gpu_parts) >= 2) {
                gpu_num = gpu_parts[2] + 0;
                gpus += gpu_num;
            }
        }
    }

    # Accumulate GPUs per user if pod is running and user is found
    if (status == "Running" && user != "") {
        total_gpus[user] += gpus;
    }
}

END {
    # Print results
    printf "%-20s %s\n", "USER", "GPUs";
    for (user in total_gpus) {
        printf "%-20s %d\n", user, total_gpus[user];
    }
}'