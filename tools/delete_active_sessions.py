#!/usr/bin/env python3
"""
One-off admin script to batch-delete active research sessions from Firestore.

Usage examples:
  # Dry-run (default): shows which sessions would be deleted
  python tools/delete_active_sessions.py

  # Delete running and pending sessions (prompt for confirmation)
  python tools/delete_active_sessions.py --yes

  # Archive then delete
  python tools/delete_active_sessions.py --archive --yes

Notes:
  - Requires google-cloud-firestore and valid Google credentials (e.g. set
    `GOOGLE_APPLICATION_CREDENTIALS` environment variable).
  - By default the script does a dry-run. Pass `--yes` to actually delete.
"""
import argparse
import sys
import os
from datetime import datetime
from typing import List

try:
    from google.cloud import firestore
except Exception:
    print("google-cloud-firestore is required. Install with: pip install google-cloud-firestore")
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-delete active research sessions from Firestore")
    parser.add_argument("--project", help="Google Cloud project id (optional)")
    parser.add_argument("--statuses", nargs="+", default=["running", "pending"],
                        help="Session statuses to target (default: running pending)")
    parser.add_argument("--archive", action="store_true", help="Archive sessions to 'archived_sessions' before deletion")
    parser.add_argument("--yes", action="store_true", help="Perform deletions (dangerous). Without --yes the script does a dry-run")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit to number of sessions to delete (0 = no limit)")
    return parser.parse_args()


def get_client(project: str = None):
    if project:
        return firestore.Client(project=project)
    return firestore.Client()


def fetch_target_sessions(db, statuses: List[str], limit: int = 0):
    col = db.collection("research_sessions")
    # Firestore 'in' accepts a list; limit optional
    q = col.where(filter=firestore.FieldFilter("status", "in", statuses))
    if limit and limit > 0:
        q = q.limit(limit)
    return list(q.stream())


def archive_doc(db, doc):
    data = doc.to_dict() or {}
    data["archived_at"] = datetime.utcnow()
    data["original_id"] = doc.id
    # Use same doc id in archive collection to avoid collisions
    archive_collection = os.getenv("ARCHIVE_COLLECTION", "archived_sessions")
    db.collection(archive_collection).document(doc.id).set(data)


def delete_doc(db, doc):
    db.collection("research_sessions").document(doc.id).delete()


def main():
    args = parse_args()
    db = get_client(args.project)

    print(f"Searching for sessions with statuses: {args.statuses}")
    docs = fetch_target_sessions(db, args.statuses, args.limit)

    if not docs:
        print("No matching sessions found.")
        return 0

    print(f"Found {len(docs)} sessions to examine:")
    for d in docs:
        dd = d.to_dict() or {}
        created = dd.get("created_at")
        print(f" - id={d.id} status={dd.get('status')} query={dd.get('query')!r} created_at={created}")

    if not args.yes:
        print("\nDry-run mode (no deletions). To perform deletions pass --yes.")
        return 0

    # Confirm once more
    confirm = input(f"Proceed to {'archive and delete' if args.archive else 'delete'} {len(docs)} sessions? Type 'DELETE' to confirm: ")
    if confirm != "DELETE":
        print("Aborting: confirmation failed.")
        return 1

    failures = 0
    for d in docs:
        try:
            if args.archive:
                archive_doc(db, d)
            delete_doc(db, d)
            print(f"Deleted session {d.id}")
        except Exception as e:
            failures += 1
            print(f"Failed to delete {d.id}: {e}")

    print(f"Done. Deleted: {len(docs)-failures}, Failures: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
