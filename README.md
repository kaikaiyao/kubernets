# Running a Kubernetes Pod on EIDF Clusters

This guide explains the process of building a Docker image on Windows, pushing it to your repository, and deploying a Kubernetes pod on the EIDF clusters. An appendix is also provided to clarify the machine relationships and SSH access setup.

---

## 1. Building and Pushing Your Docker Image (Windows / PowerShell)

1. **Build the Docker Image**  
   In the directory containing your `Dockerfile`, execute:  
   ```powershell
   docker build -t my_app .
   ```

2. **Tag and Push the Image**  
   After a successful build, tag and push the image to your Docker repository:  
   ```powershell
   docker tag my_app kaiyaoed/my_app:latest
   docker push kaiyaoed/my_app:latest
   ```

3. **Test the Built Image**  
   To verify that the image is working correctly, run:  
   ```powershell
   docker run -it --rm my_app /bin/bash
   ```

---

## 2. Deploying Your Pod on the Kubernetes Cluster

### 2.1 Creating the PersistentVolumeClaim (PVC)

Create a PVC on your target Kubernetes machine (e.g., `eidf029` or `eidf108`) by using the following command. Replace `<PVCNAME>` and `<STORAGE>` with the appropriate values (the example uses the environment variables `$USER-ws1` and `250Gi`):

```bash
kubectl create -f <(PVCNAME=$USER-ws1 STORAGE=250Gi envsubst < pvc.yml)
```

The `pvc.yml` should look like this:

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

Prepare your pod definition file `my_app.yml` (ensure the PVC name in the `claimName` field is correct):

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
          mountPath: /workspace/data  # Must match the saving path in your app
  restartPolicy: Never
```

Deploy the pod with the following command, which substitutes environment variables into the YAML file:

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

## Appendix: Machine Relationships and SSH Setup

### Overview

- **Local Machine:**  
  You operate from a Windows machine, where you build Docker images and manage SSH sessions.

- **Clusters:**  
  Two clusters are available through the school's infrastructure:
  - **EIDF Cluster 107:** Accessible via `eidf107`
  - **EIDF Cluster 029 (or infk8s):** Accessible via `eidf029`

### SSH Access Setup

SSH into the clusters using a jump host through the following commands from your Windows machine. These commands use the jump host (`eidf-gateway.epcc.ed.ac.uk`) to access the target clusters:

- **Accessing EIDF Cluster 107:**
  ```bash
  ssh -J s2470447-eidf107@eidf-gateway.epcc.ed.ac.uk s2470447-eidf107@10.24.6.77
  ```

- **Accessing EIDF Cluster 029 (infk8s):**
  ```bash
  ssh -J s2470447-infk8s@eidf-gateway.epcc.ed.ac.uk s2470447-infk8s@10.24.5.121
  ```

### Using MobaXTerm

You have configured three tabs in MobaXTerm for ease of use:
- **Windows PowerShell Tab:** For local Docker image building and testing.
- **EIDF107 Tab:** SSH session to the EIDF Cluster 107.
- **EIDF029 Tab:** SSH session to the EIDF Cluster 029 (infk8s).

When you open these tabs, they automatically connect to the corresponding machines.

### Two-Factor Authentication (TOTP)

Upon establishing an SSH session, you might be prompted to input a one-time passcode (TOTP) from Microsoft Authenticator. This extra security step ensures secure access to the clusters.
