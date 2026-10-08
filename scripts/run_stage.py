#!/usr/bin/env python3
"""Run one pipeline stage with the configured recording and output paths."""

import project_config as PC
import importlib
import sys
from capture_paths import configure,OUT

ALLOWED={'auto_features','track_fixed_features','brute_force_registration','extend_fixed_feature_observations',
         'per_frame_registration','render_distorted_views','build_per_frame_comparison_video','auto_feature_viewer','verify_fixed_feature_registration'}

def main():
    if len(sys.argv)<3:raise SystemExit('Usage: run_stage.py RUN MODULE [module arguments]')
    run,name=sys.argv[1:3]
    if name not in ALLOWED:raise ValueError('Unsupported isolated stage: '+name)
    C=configure(run)
    if name in {'auto_features','auto_feature_viewer','verify_fixed_feature_registration'}:
        import auto_features as A
        A.OUT=OUT/run/'auto_features';A.AUTO_ANNOTATIONS=C.ANNOTATIONS
    module=importlib.import_module(name)
    if name=='auto_feature_viewer':
        if hasattr(module,'AUTO_ANNOTATIONS'):module.AUTO_ANNOTATIONS=C.ANNOTATIONS
    sys.argv=[name]+sys.argv[3:]
    module.main()

if __name__=='__main__':main()
