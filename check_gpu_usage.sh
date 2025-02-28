#!/bin/bash

kubectl describe pods --all-namespaces | awk '
BEGIN {
    RS = "";  # Pods are separated by blank lines
    FS = "\n";
}

{
    status = "";
    user = "";
    gpus = 0;
    in_limits = 0;
    in_labels = 0;
    labels = "";

    for (i = 1; i <= NF; i++) {
        line = $i;

        # Capture pod status
        if (line ~ /^Status:[[:space:]]+Running/) {
            status = "Running";
        }

        # Handle multi-line Labels section
        if (line ~ /^Labels:/) {
            in_labels = 1;
            labels = substr(line, index(line, ":") + 2);
        } else if (in_labels && line ~ /^[[:space:]]/) {
            sub(/^[[:space:]]+/, "", line);
            labels = labels " " line;
        } else if (in_labels) {
            in_labels = 0;
            # Parse labels
            split(labels, label_arr, /,[[:space:]]*/);
            for (idx in label_arr) {
                if (label_arr[idx] ~ /eidf\/user=/) {
                    split(label_arr[idx], user_pair, "=");
                    user = user_pair[2];
                    break;
                }
            }
        }

        # Track GPU limits in containers
        if (line ~ /Limits:/) {
            in_limits = 1;
        }
        if (line ~ /Requests:/) {
            in_limits = 0;
        }
        if (in_limits && line ~ /nvidia.com\/gpu:/) {
            split(line, parts, /nvidia.com\/gpu:[[:space:]]+/);
            gpu_count = parts[2] + 0;
            gpus += gpu_count;
        }
    }

    if (status == "Running" && user != "") {
        total_gpus[user] += gpus;
    }
}

END {
    printf "%-20s %s\n", "USER", "GPUs";
    for (u in total_gpus) {
        printf "%-20s %d\n", u, total_gpus[u];
    }
}'