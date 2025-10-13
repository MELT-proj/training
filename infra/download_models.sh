#!/bin/bash

set -euo pipefail

dest_root=${1:-}

# Ensure a destination root is provided.
if [[ -z "$dest_root" ]]; then
	echo "Usage: $0 <destination-root>"
	exit 1
fi

# Normalize the destination root path.
dest_root=$(realpath -m "$dest_root")

# List of models to download.
models=(
	"Qwen/Qwen3-1.7B"
)

# Iterate over the models and download missing ones only.
for model in "${models[@]}"; do
	model_path="$dest_root/$model"

	if [[ -d "$model_path" ]]; then
		echo "Skipping $model (already present at $model_path)."
		continue
	fi

	mkdir -p "$(dirname "$model_path")"
	echo "Downloading $model to $model_path..."
	hf download "$model" --local-dir "$model_path"
done