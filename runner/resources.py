"""Measure memory used by this process tree, and system swap, on macOS."""

import ctypes
import os
import re
import subprocess
import time

GIB = 1024**3


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )
    ]


def descendants(rows, parent):
    owned = {parent}
    while True:
        children = {pid for pid, ppid in rows if ppid in owned}
        if children <= owned:
            return owned
        owned |= children


def memory():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    rows = [
        tuple(map(int, line.split()))
        for line in subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
    ]
    rss = footprint = 0
    for pid in descendants(rows, os.getpid()):
        usage = Usage()
        if lib.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
            continue  # The process exited after ps listed it.
        rss += usage.resident_size
        footprint += usage.phys_footprint
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    amount, unit = re.search(r"used = ([\d.]+)([KMG])", swap).groups()
    return {
        "time": time.time(),
        "rss": rss,
        "footprint": footprint,
        "swap": float(amount) * {"K": 1024, "M": 1024**2, "G": GIB}[unit],
    }


def limit_reason(sample, initial_swap):
    if max(sample["rss"], sample["footprint"]) > 42 * GIB:
        return "llama.cpp and the workers used more than 42 GiB of memory"
    if sample["swap"] - initial_swap > 4 * GIB:
        return "System swap grew by more than 4 GiB"
    return None
