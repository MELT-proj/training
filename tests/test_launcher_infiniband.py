"""The container launcher's InfiniBand and NCCL plumbing.

Each test runs the real block out of `bash/run_train_singularity.sbatch`, so a
change to the shipped logic fails here. What would silently cost every multi-node
step ~4 s: the verbs stage not reaching the image, /dev/infiniband bound again
(binds are nodev, so the devices stop opening), or NCCL_* dropped by --cleanenv.
"""

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "bash" / "run_train_singularity.sbatch"
MN5 = REPO / "infra" / "runners" / "sites" / "mn5.sh"
PRELUDE = 'container_env() { export "SINGULARITYENV_${1}=${2}"; }\nBINDS=()\n'
REPORT = 'printf "%s\\n" "${BINDS[@]}"\nenv | grep "^SINGULARITYENV_" || true\n'


def _block(start: str, end: str) -> str:
    text = SBATCH.read_text()
    i = text.index(start)
    return text[i : text.index(end, i)]


def _run(block: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", PRELUDE + block + REPORT],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
    )


def _ib(env: dict[str, str], verbs_conf: Path) -> subprocess.CompletedProcess:
    # The host's provider config lives at /etc/libibverbs.d, absent on a dev box.
    block = _block("# InfiniBand.", "BIND_ARGS=").replace("/etc/libibverbs.d", str(verbs_conf))
    return _run(block, env)


def test_a_stage_is_bound_with_its_provider_config_and_put_first_on_the_library_path(tmp_path):
    stage, conf = tmp_path / "hostib", tmp_path / "libibverbs.d"
    stage.mkdir()
    conf.mkdir()
    (stage / "libibverbs.so.1").touch()

    out = _ib({"MELT_IB_STAGE": str(stage)}, conf)

    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert f"{stage}:/opt/hostib:ro" in lines
    assert f"{conf}:{conf}:ro" in lines
    assert "SINGULARITYENV_LD_LIBRARY_PATH=/opt/hostib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" in lines


def test_an_empty_stage_runs_on_sockets(tmp_path):
    out = _ib({"MELT_IB_STAGE": ""}, tmp_path)
    assert out.returncode == 0
    assert out.stdout.strip() == ""


def test_a_stage_without_libibverbs_stops_the_job(tmp_path):
    out = _ib({"MELT_IB_STAGE": str(tmp_path)}, tmp_path)
    assert out.returncode == 2
    assert "libibverbs.so.1" in out.stdout


def test_binding_dev_infiniband_is_refused(tmp_path):
    out = _ib({"MELT_IB_STAGE": "", "MELT_EXTRA_BINDS": "/data:/data,/dev/infiniband"}, tmp_path)
    assert out.returncode == 2
    assert "nodev" in out.stdout


def test_nccl_settings_survive_cleanenv():
    block = _block("# NCCL settings", "# Bind mounts")
    out = _run(block, {"NCCL_SOCKET_NTHREADS": "8", "NCCL_DEBUG": "INFO", "OTHER": "x"})
    forwarded = set(out.stdout.splitlines())
    assert {"SINGULARITYENV_NCCL_SOCKET_NTHREADS=8", "SINGULARITYENV_NCCL_DEBUG=INFO"} <= forwarded
    assert not any("OTHER" in line for line in forwarded)


def test_mn5_turns_ib_on_by_default_and_an_empty_value_turns_it_off():
    def stage(env: dict[str, str]) -> str:
        out = subprocess.run(
            ["bash", "-c", f'source "{MN5}" >/dev/null; printf "%s" "$MELT_IB_STAGE"'],
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", **env},
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout

    assert stage({}) == "/gpfs/projects/epor48/melt-data/hostib"
    assert stage({"MELT_IB_STAGE": ""}) == ""
