"""Download pinned-checksum weights; models are not redistributed in this repo."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import project_config as PC


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = json.loads((PC.CODE_ROOT/'configs/weights.json').read_text())
    for item in entries:
        dest = PC.path(item['config_key']) if 'config_key' in item else Path(os.environ['TORCH_HOME'])/'hub/checkpoints'/item['filename']
        if dest.exists():
            if digest(dest) != item['sha256']:
                raise RuntimeError(f'Existing weight checksum differs; refusing to replace: {dest}')
            print('Verified', dest)
            continue
        if not dest.resolve().is_relative_to(PC.CODE_ROOT):
            raise ValueError(f'Download destination must be inside this repository: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix+'.partial')
        print('Downloading', item['url'], flush=True)
        try:
            urllib.request.urlretrieve(item['url'], temporary)
            if digest(temporary) != item['sha256']:
                raise RuntimeError('Downloaded checksum differs from the recorded model')
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        print('Verified', dest)


if __name__ == '__main__':
    main()
