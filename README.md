# Running a Kubernetes Pod on EIDF Clusters

This guide explains the process of building a Docker image on Windows, pushing it to your repository, deploying a Kubernetes pod on the EIDF clusters, and managing the associated PersistentVolumeClaim (PVC). An appendix is provided to clarify machine relationships and SSH access setup.

---

## 1. Building and Pushing Your Docker Image (Windows / PowerShell)

1. **Build the Docker Image**  
   In the directory containing your `Dockerfile`, execute:  
   ```powershell
   docker build -t my_app .
   ```
   Below is an updated command to reflect the latest GitHub repo all the time when rebuilding an image:
   ```powershell
   docker build --build-arg CACHEBUST=$(Get-Date -UFormat %s) -t my_app .
   ```
   
3. **Tag and Push the Image**  
   After a successful build, tag and push the image to your Docker repository:  
   ```powershell
   docker tag my_app kaiyaoed/my_app:latest
   docker push kaiyaoed/my_app:latest
   ```

4. **Test the Built Image**  
   To verify that the image works correctly, run:  
   ```powershell
   docker run -it --rm my_app /bin/bash
   ```

---

## 2. Deploying Your Pod on the Kubernetes Cluster

### 2.1 Creating the PersistentVolumeClaim (PVC)

Create a PVC on your target Kubernetes machine (e.g., `eidf029` or `eidf108`) using the following command. This example uses the environment variables `$USER-ws1` for the PVC name and `250Gi` for the storage request:

```bash
kubectl create -f <(PVCNAME=$USER-ws1 STORAGE=250Gi envsubst < pvc.yml)
```

Your `pvc.yml` should resemble:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $PVCNAME
  labels:
    eidf/user: $USER
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: $STORAGE
  storageClassName: csi-rbd-sc
```

### 2.2 Creating the Pod

Prepare your pod definition file `my_app.yml` (ensure the PVC name in the `claimName` field matches your setup):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    eidf/user: $USER
    kueue.x-k8s.io/queue-name: s2470447-infk8s-user-queue
spec:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: s2470447-infk8s-ws1
  containers:
    - name: my-app
      image: kaiyaoed/my_app:latest
      resources:
        limits:
          nvidia.com/gpu: "1"   # Request one GPU
          cpu: "1"              # Request one CPU
          memory: "4Gi"         # Request 4Gi of memory
      volumeMounts:
        - name: workspace
          mountPath: /workspace/results  # Must match the saving path in your app
  restartPolicy: Never
```

Deploy the pod by substituting environment variables into the YAML file:

```bash
WORKSPACE_PVC=$USER-ws1 envsubst < my_app.yml | kubectl create -f -
```

---

## 3. Monitoring and Managing Your Pod

- **Track Pod Status:**  
  To check the status (e.g., running, complete, or suspended), use:
  ```bash
  kubectl describe pod my-app
  ```

- **View Pod Logs:**  
  To view application logs (e.g., Python logs), run:
  ```bash
  kubectl logs my-app
  ```

- **Delete the Pod:**  
  Once finished and satisfied with the results, delete the pod with:
  ```bash
  kubectl delete pod my-app
  ```
  Confirm the deletion when prompted.

---

## 4. PVC Inspection and Cleanup

### 4.1 Inspecting the PVC

To inspect the output associated with your PVC (as per your school’s guide):

1. **Start PVC Synchronization:**  
   Bring up the synchronization process:
   ```bash
   kubectl pvcsync $USER-ws1 up
   ```
   *Wait until you see several lines of output ending with a message such as "... is listening on port ...". This may take a few seconds.*

2. **Inspect the Output:**  
   Once the sync is active, view the output file by running:
   ```bash
   kubectl exec -it $USER-ws1-rsync-backend -- ls /data
   ```

3. **Shut Down the Rsync Backend:**  
   After inspection, shut down the rsync backend pod:
   ```bash
   kubectl pvcsync $USER-ws1 down
   ```

### 4.2 Deleting the PVC

When the PVC is no longer needed, delete it using:
```bash
kubectl delete pvc $USER-ws1
```

---

## Appendix: Machine Relationships and SSH Setup

### Overview

- **Local Machine:**  
  You work from a Windows machine, where you build Docker images and manage SSH sessions.

- **Clusters:**  
  Two clusters are available through the school's infrastructure:
  - **EIDF Cluster 107:** Accessible via `eidf107`
  - **EIDF Cluster 029 (infk8s):** Accessible via `eidf029`

### SSH Access Setup

SSH into the clusters using a jump host from your Windows machine. The following commands use the jump host (`eidf-gateway.epcc.ed.ac.uk`) to access the target clusters:

- **Accessing EIDF Cluster 107:**
  ```bash
  ssh -J s2470447-eidf107@eidf-gateway.epcc.ed.ac.uk s2470447-eidf107@10.24.6.77
  ```

- **Accessing EIDF Cluster 029 (infk8s):**
  ```bash
  ssh -J s2470447-infk8s@eidf-gateway.epcc.ed.ac.uk s2470447-infk8s@10.24.5.121
  ```

### Using MobaXTerm

You have configured three tabs in MobaXTerm for convenience:
- **Windows PowerShell Tab:** For local Docker image building and testing.
- **EIDF107 Tab:** SSH session to the EIDF Cluster 107.
- **EIDF029 Tab:** SSH session to the EIDF Cluster 029 (infk8s).

Each tab automatically connects to the corresponding machine.

### Two-Factor Authentication (TOTP)

Upon establishing an SSH session, you may be prompted to enter a one-time passcode (TOTP) from Microsoft Authenticator, ensuring secure access to the clusters.

## Appendix: Other Useful Commands

Check the current resource usage of the cluster:
```bash
kubectl describe quota
```
