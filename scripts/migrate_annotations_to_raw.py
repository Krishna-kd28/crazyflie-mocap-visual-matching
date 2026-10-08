#!/usr/bin/env python3
"""Convert an annotation file from undistorted to raw pixel coordinates.

Earlier clicks were made on undistorted images and pinhole renders. The picker
now shows raw frames and distorted renders, so every click is stored in raw
pixel coordinates (coords = 'raw'); the conversion applies the supplied lens
model exactly and keeps the original value as xy_pinhole_original. Safe to run
twice: clicks already marked raw are left alone. A backup is written first.
"""
from __future__ import annotations

import project_config as PC

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_features_common as C  # noqa: E402

RAW_CONVENTION = ('Raw recorded 320x240 pixel coordinates [u, v], u right, v down, integer coordinates at '
                  'pixel centres, for real frames and for renders warped with the supplied lens model. '
                  'Geometry converts clicks to pinhole coordinates with the inverse lens model.')


def migrate(ann):
    """Convert in place; return the number of clicks converted."""
    n = 0
    for f in ann.get('features', []):
        for kind in ('render_clicks', 'real_clicks'):
            for c in f.get(kind, []):
                if c.get('coords') == 'raw':
                    continue
                pin = [float(c['xy'][0]), float(c['xy'][1])]
                raw = C.pinhole_to_raw(np.array([pin]))[0]
                c['xy_pinhole_original'] = pin
                c['xy'] = [round(float(raw[0]), 2), round(float(raw[1]), 2)]
                c['coords'] = 'raw'
                n += 1
    ann['coordinate_convention'] = RAW_CONVENTION
    ann['coords'] = 'raw'
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', type=Path, default=C.ANNOTATIONS)
    args = ap.parse_args()
    ann = json.loads(args.annotations.read_text())
    backups = C.FF_OUT / 'annotation_backups' / args.annotations.stem
    backups.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
    shutil.copy2(args.annotations, backups / f'{args.annotations.stem}_{stamp}_before_raw_migration.json')
    n = migrate(ann)
    ann['updated'] = dt.datetime.now().isoformat(timespec='seconds')
    args.annotations.write_text(json.dumps(ann, indent=2) + '\n')
    print(f'{n} clicks converted to raw coordinates -> {args.annotations}')


if __name__ == '__main__':
    main()
