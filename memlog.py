"""Process memory: log lines, and handing freed memory back to the OS.

Render's free-tier metrics are one data point per hour, far too coarse to
see what a single sync does before the instance is killed at 512MB. This
reads Linux's own counters, so it costs nothing and needs no dependency.

Why release_memory exists: a sheet sync creates ~150k small float objects
per 100-row embedding batch. Python frees them, but glibc's allocator keeps
freed memory mapped to the process (worse across the threadpool's many
malloc arenas), so RSS crept up batch after batch on Linux: a 2,000-row
sheet synced fine, a 5,000-row one got the instance killed. Locally on
Windows the same run stayed flat, which is what pointed at the allocator.
"""
import ctypes
import gc

_libc = None
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    pass  # not Linux/glibc (local Windows dev): everything here is a no-op


def limit_malloc_arenas(n: int = 2):
    """Cap glibc's per-thread malloc arenas (M_ARENA_MAX = -8). Each worker
    thread otherwise gets its own arena, each holding on to freed memory."""
    if _libc is not None:
        try:
            _libc.mallopt(-8, n)
        except Exception:
            pass


def release_memory():
    """Collect garbage, then return freed heap pages to the OS."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def mem_summary() -> str:
    try:
        fields = {}
        with open("/proc/self/status") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    fields[key] = int(value.split()[0]) // 1024  # kB -> MB
        return f"rss={fields.get('VmRSS')}MB peak={fields.get('VmHWM')}MB"
    except OSError:
        return "rss=n/a"  # not Linux (local Windows dev)
