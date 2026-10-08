"""
Manage API keys for multi-tenant access.

    python scripts/manage_keys.py create --tenant acme-corp --label "Acme prod key"
    python scripts/manage_keys.py list
    python scripts/manage_keys.py list --tenant acme-corp
    python scripts/manage_keys.py revoke --hash-prefix a1b2c3d4

The plaintext key is shown EXACTLY ONCE, at creation. Only its SHA-256 hash
is stored -- there is no "show key again" command, by design (the same
reason you can't ask a website to show you your password back).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a new API key for a tenant")
    create.add_argument("--tenant", required=True, help="Tenant id (your choice of identifier, e.g. a customer name)")
    create.add_argument("--label", required=True, help="Human-readable label for this key (e.g. 'prod', 'alice-laptop')")

    lst = sub.add_parser("list", help="List keys (shows hash prefix, tenant, label -- never the plaintext key)")
    lst.add_argument("--tenant", default=None)

    revoke = sub.add_parser("revoke", help="Revoke a key by its hash prefix (see `list` output)")
    revoke.add_argument("--hash-prefix", required=True)

    args = ap.parse_args()

    from app.auth import generate_api_key
    from app.config import get_settings
    from app.repositories.tenant_repository import TenantRepository

    settings = get_settings()
    repo = TenantRepository(settings.tenant_db_path)

    if args.command == "create":
        key = generate_api_key()
        repo.create_key(key, tenant_id=args.tenant, label=args.label)
        print(f"Created a key for tenant '{args.tenant}':\n")
        print(f"  {key}\n")
        print("This is shown ONCE. Store it securely now -- it cannot be retrieved again.")
        print(f'Use it as:  Authorization: Bearer {key}')
        return 0

    if args.command == "list":
        keys = repo.list_keys(args.tenant)
        if not keys:
            print("No keys found." + (f" (tenant: {args.tenant})" if args.tenant else ""))
            return 0
        print(f"{'hash prefix':<14} {'tenant':<20} {'label':<30} {'status':<10}")
        print("-" * 76)
        for k in keys:
            status = "REVOKED" if k.revoked else "active"
            print(f"{k.key_hash[:12]:<14} {k.tenant_id:<20} {k.label[:30]:<30} {status:<10}")
        return 0

    if args.command == "revoke":
        count = repo.revoke(args.hash_prefix)
        if count == 0:
            print(f"No key matched prefix '{args.hash_prefix}'.")
            return 1
        print(f"Revoked {count} key(s) matching prefix '{args.hash_prefix}'.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
