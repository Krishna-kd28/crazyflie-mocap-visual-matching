#!/usr/bin/env python3
"""Read-only checks for source closure, links, and copied-file provenance."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path,help='Optional original workspace for unchanged-source verification')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    failures=[];checks={}
    scripts=list((ROOT/'scripts').glob('*.py'))
    for p in scripts:
        ast.parse(p.read_text())
        if '/home/krishna-dubey/' in p.read_text():failures.append(f'Author-specific path in {p.relative_to(ROOT)}')
    checks['python_files_parsed']=len(scripts)
    links=0
    for p in [ROOT/'README.md',*ROOT.glob('docs/**/*.md'),*ROOT.glob('provenance/*.md'),*ROOT.glob('verification/*.md')]:
        for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in ref or ref.startswith('#'):continue
            target=(p.parent/ref.split('#')[0]).resolve();links+=1
            if not target.exists():failures.append(f'Broken link {p.relative_to(ROOT)} -> {ref}')
    checks['local_document_links']=links
    manifest=json.loads((ROOT/'provenance/source_files.json').read_text())['files']
    verified=0
    for row in manifest:
        dest=ROOT/row['destination']
        if not dest.is_file():failures.append(f'Missing copied file {row["destination"]}');continue
        if sha(dest)!=row['packaged_sha256']:
            failures.append(f'File differs from its recorded hash: {row["destination"]}')
        else:verified+=1
    checks['attributed_files']=len(manifest);checks['verified_file_hashes']=verified
    if args.original_root:
        protected=json.loads((ROOT/'verification/original_files_before.json').read_text())
        changed=[row['path'] for row in protected if not (args.original_root/row['path']).is_file() or sha(args.original_root/row['path'])!=row['sha256']]
        checks['original_files_checked']=len(protected)
        checks['original_files_changed']=changed
        if changed:failures.append('Original files changed since packaging snapshot')
    result=dict(status='failed' if failures else 'passed',**checks,failures=failures)
    print(json.dumps(result,indent=2))
    if args.out:
        dest=args.out.resolve()
        if not dest.is_relative_to(ROOT):raise ValueError('Report must stay inside this package')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
