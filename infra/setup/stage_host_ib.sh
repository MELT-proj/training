#!/bin/bash
# Copy the host's InfiniBand verbs userspace into a directory the container can
# load it from (MELT_IB_STAGE in infra/runners/sites/*.sh).
#
#   infra/setup/stage_host_ib.sh <dest_dir>
#
# The image ships no rdma-core, so without these NCCL cannot use the verbs devices
# and falls back to TCP sockets between nodes. Taking the HOST's libraries, not a
# packaged copy, keeps the provider matched to the host's kernel driver. Run it on
# a node with the same rdma-core as the compute nodes (on MN5 the login nodes match;
# the md5 sums printed at the end make that checkable), and again after an OS update.
#
# bash/run_train_singularity.sbatch binds <dest_dir> at /opt/hostib, puts it first
# on LD_LIBRARY_PATH and binds /etc/libibverbs.d, which names the mlx5 provider.
set -euo pipefail

DEST="${1:?usage: $0 <dest_dir>}"
LIBDIR="${IB_HOST_LIBDIR:-/usr/lib64}"

# libibverbs and its mlx5 provider (also loaded by its rdmav name, which is what
# libibverbs.d resolves to), libibumad, and libnl, which the provider links against.
LIBS=(libibverbs.so.1 libibumad.so.3 libmlx5.so.1 libnl-3.so.200 libnl-route-3.so.200)

mkdir -p "$DEST"
for lib in "${LIBS[@]}"; do
    [[ -e "${LIBDIR}/${lib}" ]] || { echo "ERROR: ${LIBDIR}/${lib} not found" >&2; exit 1; }
    cp -L "${LIBDIR}/${lib}" "${DEST}/${lib}"
done
shopt -s nullglob
providers=("${LIBDIR}"/libibverbs/libmlx5-rdmav*.so)
[[ ${#providers[@]} -gt 0 ]] || { echo "ERROR: no ${LIBDIR}/libibverbs/libmlx5-rdmav*.so" >&2; exit 1; }
cp -L "${providers[@]}" "$DEST/"
chmod 0755 "$DEST"/*

echo "Staged into ${DEST}:"
(cd "$DEST" && md5sum ./*)
