#!/usr/bin/env python3
"""Archive Git-tracked source only; never bundle weights/data/workspaces."""
import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    files=sorted(p for p in files if p)
    if not files:raise RuntimeError('Initialize and add the source files to Git first')
    out=ROOT/'dist';out.mkdir(exist_ok=True)
    dest=out/'crazyflie-mocap-visual-matching.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in files:
            p=ROOT/name
            if p.is_symlink():raise ValueError(f'Source archive cannot depend on a symlink: {name}')
            if name.startswith(('workspace','data/','assets/','dist/','.git/')) or name.startswith('configs/local'):
                raise ValueError(f'Excluded runtime file is unexpectedly tracked: {name}')
            info=zipfile.ZipInfo('crazyflie-mocap-visual-matching/'+name,date_time=(2026,10,7,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix('.zip.sha256').write_text(f'{digest}  {dest.name}\n')
    print(f'{len(files)} files, {dest.stat().st_size:,} bytes: {dest}')


if __name__=='__main__':main()
