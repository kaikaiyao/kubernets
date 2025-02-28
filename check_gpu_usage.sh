#!/bin/bash

# Process the output of kubectl describe pods to extract GPU usage per user
kubectl describe pods | awk '
# When a new pod description starts
/^Name: / {
    # If the previous pod was running and has a user, output its user and GPU count
    if (current_pod != "" && is_running && user != "") {
        print user, gpus
    }
    # Reset variables for the new pod
    current_pod = $2
    is_running = 0
    user = ""
    gpus = 0
}

# Check if the pod is currently running
/^Status: Running/ {
    is_running = 1
}

# Extract the user from lines containing the eidf/user= label
/eidf\/user=/ {
    match($0, /eidf\/user=([^ ]+)/, a)
    user = a[1]
}

# Sum GPU requests from lines specifying nvidia.com/gpu under Requests
/^      nvidia.com\/gpu: / {
    gpus += $2
}

# Handle the last pod at the end of input
END {
    if (current_pod != "" && is_running && user != "") {
        print user, gpus
    }
}
' | awk '{
    # Sum the GPUs for each user across all their running pods
    s[$1] += $2
} END {
    # Output each user and their total GPU usage
    for (u in s) print u, s[u]
}'