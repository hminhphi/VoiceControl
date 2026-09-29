import builtins
import os
import re
import sys
from datetime import datetime

_ORIGINAL_PRINT = builtins.print
_INSTALLED = False
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def _env_bool(name, default=True):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _timestamp():
    now = datetime.now().astimezone()
    return f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}{now:%z}"


def _prefix_lines(text):
    if not text:
        return text

    stamp = _timestamp()
    prefixed = []
    for line in text.splitlines(keepends=True):
        if not line.strip() or _TIMESTAMP_RE.match(line):
            prefixed.append(line)
        else:
            prefixed.append(f"{stamp} {line}")
    return "".join(prefixed)


def timestamped_print(*args, **kwargs):
    file = kwargs.get("file")
    if file not in (None, sys.stdout, sys.stderr):
        return _ORIGINAL_PRINT(*args, **kwargs)

    sep = kwargs.pop("sep", " ")
    text = sep.join(str(arg) for arg in args)
    return _ORIGINAL_PRINT(_prefix_lines(text), sep="", **kwargs)


def install_timestamped_print():
    global _INSTALLED
    if _INSTALLED or not _env_bool("VOICE_LOG_TIMESTAMPS", True):
        return

    builtins.print = timestamped_print
    _INSTALLED = True
