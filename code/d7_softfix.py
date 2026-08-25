#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate *_soft.npy artifacts from the intact *_prof.npy profiles.

The D6/D7/D7b inference scripts all had the same soft-argmax bug:
    soft = numerator / pred.sum(-1, keepdim=True)   # (R,) / (R,1) -> (R,R)
    save soft[0]                                     # row 0: ray-0 num / every ray's mass
so every saved _soft.npy held garbage (verified: max |saved - correct| up to
13.8 bins on the intact 12/60/60 objects). The _prof.npy files are correct
((R,96)), so the correct soft depth is exactly recoverable:
    soft = (prof * bins).sum(-1) / prof.sum(-1)
Usage: python d7_softfix.py <pred_dir>
"""
import os
import sys

import numpy as np

N_BINS = 96
AXIS = np.arange(N_BINS, dtype=np.float32)


def main():
    pred_dir = sys.argv[1]
    objs = sorted({f[:-9] for f in os.listdir(pred_dir) if f.endswith("_prof.npy")})
    fixed, skipped = 0, 0
    for n in objs:
        p = os.path.join(pred_dir, n + "_prof.npy")
        prof = np.load(p).astype(np.float32)
        if prof.ndim != 2 or prof.shape[1] != N_BINS:
            print("skip %s (prof shape %s)" % (n, prof.shape))
            skipped += 1
            continue
        soft = (prof * AXIS[None, :]).sum(-1) / (prof.sum(-1) + 1e-6)
        np.save(os.path.join(pred_dir, n + "_soft.npy"), soft.astype(np.float32))
        fixed += 1
    print("fixed %d objects in %s (skipped %d)" % (fixed, pred_dir, skipped))


if __name__ == "__main__":
    main()
