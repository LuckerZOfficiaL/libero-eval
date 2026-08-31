#!/usr/bin/env python3
"""One-shot relocation for a new host — no root, no mount at a specific path.

The project was written with two absolute prefixes baked in:
    /home/lzhou/libero-eval      (repo root, in ~20 driver files)
    /media/pinas                 (NAS data root, in ~10 places AND in the 76 paths inside plan.json)

Re-creating `/media/pinas` on a new machine needs sudo. Instead, run this once after copying the repo:

    export LIBERO_DATA_ROOT=/wherever/the/nas/is        # only if it is not auto-detected
    python sweep/relocate.py                            # dry run: shows every change
    python sweep/relocate.py --apply                    # rewrites in place, keeps .bak copies

It rewrites both prefixes to what `sweep/paths.py` resolves on THIS machine, and is idempotent — safe
to run again after any later rsync from the old host.

Note the eval results themselves (`results/merge/*.jsonl`) contain no paths, so nothing there needs
touching and resume-by-record keeps working.
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paths as P

OLD_REPO = "/home/lzhou/libero-eval"
OLD_DATA = "/media/pinas"
TARGETS = ["sweep/*.py", "scripts/*.py", "sweep/*.json", "configs/*.yaml"]


def rewrites():
    out = []
    if str(P.REPO) != OLD_REPO:
        out.append((OLD_REPO, str(P.REPO)))
    if str(P.DATA_ROOT) != OLD_DATA:
        out.append((OLD_DATA, str(P.DATA_ROOT)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    a = ap.parse_args()

    print(f"REPO      {OLD_REPO}  ->  {P.REPO}")
    print(f"DATA_ROOT {OLD_DATA}  ->  {P.DATA_ROOT}\n")
    subs = rewrites()
    if not subs:
        print("Nothing to do — both prefixes already match this machine.")
        return 0
    if not P.check(verbose=False):
        print("WARNING: required data is not reachable at the resolved DATA_ROOT.")
        print("         Set LIBERO_DATA_ROOT and re-run. Details:\n")
        P.check()
        print()

    files, total = [], 0
    for pat in TARGETS:
        for f in sorted(P.REPO.glob(pat)):
            if f.name == "relocate.py":
                continue
            try:
                txt = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            n = sum(txt.count(o) for o, _ in subs)
            if n:
                files.append((f, txt, n))
                total += n
    for f, _, n in files:
        print(f"  {n:>4} occurrence(s)  {f.relative_to(P.REPO)}")
    print(f"\n{total} replacement(s) across {len(files)} file(s)")

    if not a.apply:
        print("\nDry run. Re-run with --apply to write (originals kept as *.bak).")
        return 0
    for f, txt, _ in files:
        shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
        for o, new in subs:
            txt = txt.replace(o, new)
        f.write_text(txt)
    print("\nApplied. Verifying...")
    ok = P.check()
    plan = P.REPO / "sweep/plan.json"
    if plan.exists():
        pl = json.load(open(plan))["adapters"]
        seen = sum(pathlib.Path(x["adapter_file"]).exists() for x in pl)
        print(f"  [{'ok  ' if seen == len(pl) else 'FAIL'}] plan.json adapters resolve: {seen}/{len(pl)}")
        ok &= seen == len(pl)
    print("\n" + ("Relocation complete." if ok else "Relocation incomplete — see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
