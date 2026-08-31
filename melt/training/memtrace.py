"""Opt-in memory diagnostics for training worker processes.

Every instrument here is disabled unless its environment variable is set, and
costs close to nothing when off. Not tied to any particular dataloader
implementation, so it can be pointed at any worker process.

Stacked cheapest to most expensive:

- :func:`start_worker_memstats` -- O(1) counters (``sys.getallocatedblocks()``,
  glibc ``mallinfo2``). Safe to leave running for the life of a job.
- :func:`heap_breakdown` -- a full GC walk, grouped by type. Paced separately
  from the O(1) counters and coarser, since it is far more expensive.
- :func:`start_worker_tracemalloc` -- line-level allocation attribution, at
  real per-allocation overhead. Meant to be turned on briefly to find a
  culprit, not left on.
"""

import ctypes
import os
import sys
import threading
import time
import warnings
from typing import Any

from ..logging_utils import get_logger

logger = get_logger(__name__)


def start_worker_tracemalloc(worker_id: int) -> None:
    """Periodically dump the process's top Python allocation sites. Opt-in.

    Set ``MELT_WORKER_TRACEMALLOC`` to a dump interval in seconds.

    tracemalloc only sees allocations routed through CPython's own allocator,
    so it is blind to numpy/torch buffers allocated below it -- a "such-and-
    such% in this open() call" reading from it can be misleading on its own
    and should be cross-checked against :func:`start_worker_memstats` and
    :func:`heap_breakdown`, which see the whole process.
    """
    interval = os.environ.get("MELT_WORKER_TRACEMALLOC", "")
    if not interval or interval == "0":
        return

    try:
        every = float(interval)
    except ValueError:
        logger.warning(
            "MELT_WORKER_TRACEMALLOC=%r is not a number; worker tracemalloc not started.",
            interval,
        )
        return

    import tracemalloc

    # Frame depth is the throughput/detail trade-off, and it is severe: deep
    # frames (10+) can slow a step by an order of magnitude or more, which
    # makes the traced run useless for measuring a growth *rate* even while it
    # correctly identifies the allocating line. Identify the line once with
    # depth, then drop to 1-2 frames for the same attribution at a fraction of
    # the cost. Default low and let the caller opt into depth.
    depth = int(os.environ.get("MELT_WORKER_TRACEMALLOC_FRAMES", "2"))
    tracemalloc.start(depth)
    logger.warning(
        "[wtrace] worker %d — tracemalloc started (%d frames), dumping every %.0fs", worker_id, depth, every
    )

    def _dump() -> None:
        baseline = tracemalloc.take_snapshot()
        while True:
            time.sleep(every)
            try:
                snap = tracemalloc.take_snapshot()
                # Compare against the first snapshot rather than the previous
                # one: a slow leak is a cumulative signal, while consecutive
                # diffs are mostly per-batch churn.
                stats = snap.compare_to(baseline, "traceback")[:5]
                total = sum(s.size for s in snap.statistics("filename")) / 1048576
                logger.warning(
                    "[wtrace] worker %d — python-tracked total %.0f MB; top growth:",
                    worker_id, total,
                )
                for i, stat in enumerate(stats, 1):
                    where = stat.traceback.format()[-3:]
                    logger.warning(
                        "[wtrace]   #%d +%.1f MB (%d blocks) %s",
                        i, stat.size_diff / 1048576, stat.count_diff,
                        " | ".join(w.strip() for w in where),
                    )
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill the run
                logger.warning("[wtrace] worker %d — dump failed: %s", worker_id, exc)

    threading.Thread(target=_dump, daemon=True, name=f"memtrace-{worker_id}").start()


class _MallInfo2(ctypes.Structure):
    """glibc's ``struct mallinfo2`` (malloc.h). All fields are ``size_t`` since
    glibc 2.33; the older ``mallinfo`` used ``int`` and silently wrapped past
    4 GB, which is useless here -- the numbers we care about are tens of GB."""

    _fields_ = [
        ("arena", ctypes.c_size_t),  # bytes obtained from sbrk (the main heap)
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),  # count of mmap'd regions
        ("hblkhd", ctypes.c_size_t),  # bytes in mmap'd regions
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),  # bytes currently allocated (in use)
        ("fordblks", ctypes.c_size_t),  # bytes free inside the heap
        ("keepcost", ctypes.c_size_t),  # releasable bytes at the top of the heap
    ]


# The pid that already has a sampler thread, not a bool: a DataLoader worker is
# *forked*, so it inherits the parent's module globals but not the parent's
# threads. A plain flag would be True in every worker and no worker would ever
# start its own sampler -- exactly the processes being measured.
_memstats_pid: int | None = None


def start_worker_memstats(worker_id: int) -> None:
    """Periodically log where a process's memory actually lives. Opt-in, ~free.

    Set ``MELT_WORKER_MEMSTATS`` to an interval in seconds. ``worker_id`` is -1
    for a caller with no worker subprocess concept (e.g. ``num_workers=0``).

    Three O(1) counters separate the families of memory growth a training
    process can exhibit, each with a different fix:

      - ``sys.getallocatedblocks()`` -- live CPython allocator blocks. Grows in
        step with RSS only if Python *objects* are accumulating, i.e. something
        is holding references. Then the fix is to find the container.
      - ``mallinfo2().uordblks`` -- bytes malloc currently considers in use.
        Grows while the block count stays flat when the growth is raw buffers
        (numpy/torch allocate below CPython's allocator, so tracemalloc is
        blind to them and can report a misleadingly clean picture).
      - ``mallinfo2().arena`` vs ``uordblks`` -- if in-use is flat while arena
        keeps growing, nothing is leaking at all: memory is freed but never
        returned to the OS. That is heap fragmentation, and the fix is an
        allocator knob (``MALLOC_MMAP_THRESHOLD_``, ``malloc_trim``), not a
        code change. ``hblkhd`` distinguishes the usual cause: glibc raises its
        mmap threshold dynamically up to 32 MB, after which large variable-size
        buffers stop being mmap'd and start fragmenting the heap instead.
    """
    global _memstats_pid
    interval = os.environ.get("MELT_WORKER_MEMSTATS", "")
    if not interval or interval == "0" or _memstats_pid == os.getpid():
        return

    try:
        every = float(interval)
    except ValueError:
        logger.warning(
            "MELT_WORKER_MEMSTATS=%r is not a number; memstats not started.", interval
        )
        return

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mallinfo2.restype = _MallInfo2
        libc.mallinfo2()  # probe now: glibc < 2.33 has no such symbol
    except (OSError, AttributeError) as exc:
        logger.warning("[memstats] mallinfo2 unavailable (%s); reporting RSS only.", exc)
        libc = None

    _memstats_pid = os.getpid()
    who = "rank" if worker_id < 0 else f"worker {worker_id}"
    logger.warning("[memstats] %s — sampling every %.0fs (pid %d)", who, every, os.getpid())

    def _rss_anon_mb() -> float:
        try:
            with open(f"/proc/{os.getpid()}/status") as fh:
                for line in fh:
                    if line.startswith("RssAnon:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0

    # A full GC walk is far more expensive than the O(1) counters, so it gets
    # its own (coarser) interval rather than running on every sample.
    heap_every = float(os.environ.get("MELT_WORKER_HEAPDUMP", "0") or 0)
    t_heap = [time.time()]

    def _dump() -> None:
        mb = 1048576.0
        while True:
            time.sleep(every)
            try:
                fields = [
                    f"rss_anon={_rss_anon_mb():.0f}MB",
                    # O(1). Deliberately not len(gc.get_objects()), which would
                    # materialise a list of every tracked object -- hundreds of
                    # MB of transient allocation inside the very process whose
                    # memory we are trying to measure.
                    f"py_blocks={sys.getallocatedblocks()}",
                ]
                if heap_every and (time.time() - t_heap[0]) >= heap_every:
                    t_heap[0] = time.time()
                    logger.warning("[memstats] %s — heap: %s", who, heap_breakdown())
                if libc is not None:
                    mi = libc.mallinfo2()
                    fields += [
                        f"arena={mi.arena / mb:.0f}MB",
                        f"in_use={mi.uordblks / mb:.0f}MB",
                        f"heap_free={mi.fordblks / mb:.0f}MB",
                        f"mmapped={mi.hblkhd / mb:.0f}MB",
                        f"mmap_blocks={mi.hblks}",
                        f"trimmable={mi.keepcost / mb:.0f}MB",
                    ]
                logger.warning("[memstats] %s — %s", who, " ".join(fields))
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill the run
                logger.warning("[memstats] %s — sample failed: %s", who, exc)

    threading.Thread(target=_dump, daemon=True, name=f"memstats-{worker_id}").start()


def heap_breakdown(top_n: int = 12) -> str:
    """Group every live Python object by type and report the biggest holders.

    ``mallinfo2`` says *how much* memory is held and in what shape; this says
    *what* is holding it. numpy arrays and torch tensors get their real payload
    size (``nbytes``), since ``sys.getsizeof`` on either reports only the small
    Python wrapper and would hide exactly the allocations worth finding.

    Walking the GC costs a list of one pointer per tracked object, which is
    affordable as a periodic diagnostic but real -- unlike ``tracemalloc``,
    which taxes every allocation and can slow a run by an order of magnitude.
    """
    import gc
    from collections import defaultdict

    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    def _size(obj: Any) -> int:
        nbytes = getattr(obj, "nbytes", None)
        if isinstance(nbytes, int):
            return nbytes  # numpy ndarray and anything else exposing a payload
        if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
            return obj.element_size() * obj.nelement()  # torch.Tensor
        return sys.getsizeof(obj)

    def _record(obj: Any) -> None:
        try:
            cls = type(obj)
            totals[f"{cls.__module__}.{cls.__qualname__}"] += _size(obj)
            counts[f"{cls.__module__}.{cls.__qualname__}"] += 1
        except Exception:  # noqa: BLE001 - a diagnostic must not kill the run
            pass

    # gc.get_objects() returns only GC-*tracked* objects, and a numeric numpy
    # array holds no references so it is not tracked -- a tracked-only walk
    # would miss exactly the allocations worth finding. One hop through
    # gc.get_referents() reaches them, since an array that is alive at all is
    # reachable from some tracked list, dict or instance __dict__.
    seen: set[int] = set()
    # Attribute probing trips deprecation warnings on some module-level objects
    # (e.g. torch.distributed.reduce_op); a diagnostic should not spam the log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracked = gc.get_objects()
        try:
            for obj in tracked:
                if id(obj) not in seen:
                    seen.add(id(obj))
                    _record(obj)
                try:
                    referents = gc.get_referents(obj)
                except Exception:  # noqa: BLE001
                    continue
                for ref in referents:
                    if id(ref) not in seen:
                        seen.add(id(ref))
                        _record(ref)
        finally:
            del tracked, seen

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    parts = [
        f"{name}={total / 1048576:.0f}MB/{counts[name]}" for name, total in ranked
    ]
    return " ".join(parts)
