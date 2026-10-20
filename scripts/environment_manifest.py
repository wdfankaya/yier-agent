"""Capture reproducible Python/dependency/source identifiers without environment secrets."""
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path


def manifest():
    root=Path(__file__).resolve().parents[1]
    lock=root/'requirements-tested.lock.txt'
    normalize=lambda s:re.sub(r'[-_.]+','-',s).lower()
    installed={normalize(d.metadata['Name']):d.version for d in importlib.metadata.distributions()}
    expected={normalize(line.split('==')[0]):line.split('==')[1] for line in lock.read_text(encoding='utf-8').splitlines() if '==' in line}
    mismatches=[{'package':name,'expected':version,'actual':installed.get(name)} for name,version in expected.items() if installed.get(name)!=version]
    files=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard'],cwd=root,text=True,capture_output=True,check=True).stdout.splitlines()
    digest=hashlib.sha256()
    for name in sorted(set(f for f in files if f.endswith('.py') or f=='requirements-tested.lock.txt')):
        path=root/name
        if path.is_file(): digest.update(name.encode()); digest.update(path.read_bytes())
    python=platform.python_version()
    return {'python':python,'python_executable':sys.executable,'platform':platform.platform(),
        'expected_python':(root/'.python-version').read_text().strip(),'python_matches':python==(root/'.python-version').read_text().strip(),
        'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'locked_packages':len(expected),
        'dependencies_match':not mismatches,'dependency_mismatches':mismatches,'source_sha256':digest.hexdigest()}

if __name__=='__main__':
    import json
    print(json.dumps(manifest(),ensure_ascii=False,indent=2))
