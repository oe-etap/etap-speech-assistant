#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Makes the package runnable: `python -m evaluation --run-dir ...`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
