"""Current and peak memory of this process, for log lines.

Render's free-tier metrics are one data point per hour, far too coarse to
see what a single sync does before the instance is killed at 512MB. This
reads Linux's own counters, so it costs nothing and needs no dependency.
"""


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
