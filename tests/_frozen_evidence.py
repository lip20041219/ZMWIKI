"""Line-ending-independent checks for frozen repository evidence.

Git stores these files with LF, but a Windows checkout may materialize them
with CRLF. Frozen SHA-256 checks therefore compare the content under either
canonical line-ending form; this still detects any non-line-ending edit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["frozen_digest_matches", "normalized_lf_bytes"]


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def frozen_digest_matches(path: str | Path, expected: str) -> bool:
    raw = Path(path).read_bytes()
    lf = _lf(raw)
    crlf = lf.replace(b"\n", b"\r\n")
    variants = {raw, lf, crlf}
    return expected in {hashlib.sha256(data).hexdigest() for data in variants}


def normalized_lf_bytes(path: str | Path) -> bytes:
    return _lf(Path(path).read_bytes())
