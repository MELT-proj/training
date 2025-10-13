#!/bin/bash

set -euo pipefail

archive_path=${1:-}
parent_dir=${2:-}

# Ensure a tarball path was supplied.
if [[ -z "$archive_path" ]]; then
	echo "Usage: $0 <path-to-conda-env-tar.gz> <parent-directory>"
	exit 1
fi

# Ensure an extraction parent directory was supplied.
if [[ -z "$parent_dir" ]]; then
	echo "Usage: $0 <path-to-conda-env-tar.gz> <parent-directory>"
	exit 1
fi

# Confirm the tarball exists before continuing.
if [[ ! -f "$archive_path" ]]; then
	echo "Archive not found: $archive_path"
	exit 1
fi

# Compute the archive file name to derive the environment name.
archive_name=$(basename "$archive_path")
env_name=${archive_name%.tar.gz}
if [[ "$env_name" == "$archive_name" ]]; then
	env_name=${archive_name%.tgz}
fi

# Abort if the environment name could not be parsed.
if [[ -z "$env_name" || "$env_name" == "$archive_name" ]]; then
	echo "Unable to derive environment name from archive: $archive_name"
	exit 1
fi

# Normalize the parent directory path.
parent_dir=$(realpath -m "$parent_dir")

# Resolve the final environment path inside the parent directory.
env_path="$parent_dir/$env_name"

# Inform the user about the extraction plan and ask for confirmation.
echo "About to extract '$archive_name' into '$env_path' and activate the contained conda environment."
read -r -p "Proceed? [y/N] " response
case "$response" in
	[yY][eE][sS]|[yY])
		;;
	*)
		echo "Aborting at user request."
		exit 0
		;;
esac

# Create the parent directory if necessary.
mkdir -p "$parent_dir"

# Check if the environment directory already exists.
if [[ -d "$env_path" ]]; then
    echo "Warning: Environment directory already exists: $env_path"
    read -r -p "Do you want to remove it and continue? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "Removing existing environment..."
            rm -rf "$env_path"
            ;;
        *)
            echo "Aborting at user request."
            exit 0
            ;;
    esac
fi
mkdir -p "$env_path"

# Extract the tarball content into the target environment directory.
tar -xzf "$archive_path" -C "$env_path"

# Activate the freshly unpacked conda environment.
source "$env_path/bin/activate"

# Rehydrate the environment paths to match the current location.
conda-unpack

# Perform a quick sanity check of the PyTorch installation.
echo "Verifying PyTorch installation..."
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"