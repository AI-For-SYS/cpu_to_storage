#!/bin/bash

POD_NAME="io-benchmark-pod"
NAMESPACE="offloading"
DEST_PATH="/workspace"

# Check pod status
# kubectl get pod "$POD_NAME" -n "$NAMESPACE" || exit 1

# Create destination directory
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- mkdir -p "$DEST_PATH"

# Copy files
kubectl cp compare_file_operations.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp build.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp setup.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp benchmark_cpp_utils "$NAMESPACE/$POD_NAME:$DEST_PATH/"

echo "Files copied to $DEST_PATH in pod $POD_NAME"
