#!/usr/bin/env python3
"""Compatibility entry point. Prefer `patentagent extract`."""

from patentagent.postprocess import _looks_markush, _rdkit_canonical  # noqa: F401


if __name__ == "__main__":
    import sys
    from patentagent.cli import main

    raise SystemExit(main(["extract", *sys.argv[1:]]))
