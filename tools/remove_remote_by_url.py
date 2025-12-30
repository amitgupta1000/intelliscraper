#!/usr/bin/env python
import subprocess, sys
if len(sys.argv) < 2:
    print('Usage: remove_remote_by_url.py <remote-url>')
    sys.exit(2)
url_to_remove = sys.argv[1].rstrip('/')
try:
    remotes = subprocess.check_output(['git','remote'], text=True).strip().split() or []
except subprocess.CalledProcessError as e:
    print('Failed to list remotes:', e)
    sys.exit(1)
removed = []
for r in remotes:
    try:
        u = subprocess.check_output(['git','remote','get-url',r], text=True).strip().rstrip('/')
    except subprocess.CalledProcessError:
        continue
    print(f"Remote {r} -> {u}")
    if u == url_to_remove:
        try:
            subprocess.check_call(['git','remote','remove', r])
            print(f"Removed remote: {r}")
            removed.append(r)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove remote {r}: {e}")
if not removed:
    print('No matching remote found to remove.')
else:
    print('Removed remotes:', ','.join(removed))
# Show remaining remotes
try:
    print('\nRemaining remotes:')
    print(subprocess.check_output(['git','remote','-v'], text=True))
except Exception:
    pass
