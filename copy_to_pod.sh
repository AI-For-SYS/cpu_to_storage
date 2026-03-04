#!/bin/bash

POD_NAME="io-benchmark-pod"
NAMESPACE="rotem"
DEST_PATH="/workspace"
CONTAINER_NAME="python-benchmarking"
LOCAL_RESULTS_DIR="./results"

# Function to copy results back from pod
copy_results_from_pod() {
    echo "Copying results from pod to local machine..."
    
    # Create local results directory if it doesn't exist
    mkdir -p "$LOCAL_RESULTS_DIR"
    
    # Copy results directory from pod
    kubectl cp "$NAMESPACE/$POD_NAME:$DEST_PATH/results" "$LOCAL_RESULTS_DIR" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "Results copied successfully to $LOCAL_RESULTS_DIR"
    else
        echo "Warning: No results found or error copying results"
    fi
}

# Check if we're only copying results back
if [ "$1" == "--get-results" ]; then
    copy_results_from_pod
    exit 0
fi

# Check pod status
# kubectl get pod "$POD_NAME" -n "$NAMESPACE" || exit 1

# Create destination directory
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- mkdir -p "$DEST_PATH"
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- mkdir -p "$DEST_PATH/results"

# Copy files to pod
echo "Copying files to pod..."
kubectl cp compare_file_operations.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp setup.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp checkpoints_utils.py "$NAMESPACE/$POD_NAME:$DEST_PATH/"
kubectl cp backends "$NAMESPACE/$POD_NAME:$DEST_PATH/"

echo "Files copied to $DEST_PATH in pod $POD_NAME"
echo ""
echo "To copy results back after benchmark completes, run:"
echo "  ./copy_to_pod.sh --get-results"

