"""Utility functions for policy training"""

from pathlib import Path
import numpy as np
import contextlib
import time
import sys


def colorstr(*input):
    """Color a string for terminal output"""
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


class Stopwatch(contextlib.ContextDecorator):
    """Stopwatch for timing code blocks"""

    def __init__(self, t=0.0):
        self.t = t
        self.avg_dt = 0
        self.avg_per_s = 0
        self.count = 0
        self.dt = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.dt = time.time() - self.start
        self.t += self.dt
        self.count += 1
        self.avg_dt += (self.dt - self.avg_dt) / self.count
        self.avg_per_s += (1 / self.dt - self.avg_per_s) / self.count


def second2str(second):
    """Convert seconds to HH:MM:SS format"""
    s = int(second)
    m = int(second // 60)
    h = int(m // 60)
    ret = ""
    if h:
        ret += f"{h:02}:"
        m = int(m % 60)
    s = int(s % 60)
    ret += f"{m:02}:{s:02}"
    return ret


class Config:
    """Base configuration class"""

    def __iter__(self):
        for name in dir(self):
            if not name.startswith("_"):
                yield name, getattr(self, name)

    def __str__(self):
        return str({k: v for k, v in self})
