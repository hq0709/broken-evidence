"""
Regenerate `cfqa_*/seg_cache/` — the TotalSegmentator anatomy masks the repo
excludes (593 files / 544 MB) and every image condition of the identification
control needs to draw the target outline.

This is deliberately a thin driver and not a call into `run_pipeline.py`.
`run_pipeline.process_one` regenerates the QA corpus as well, and the corpus is
already fixed and committed; rerunning its generation would risk writing a
different corpus over the one every reported number refers to. The one thing
needed here is the segmentation step, so the `totalsegmentator()` call below is
character-for-character the one in `run_pipeline.segment`, including `ml=True`
(single multilabel file, which is what `load_ras(seg) == label_id` expects).

TotalSegmentator brings its own torch and its own weights, so this runs in a
separate environment from the evaluation code; keeping it a standalone file with
no repo imports is what makes that possible.

Resumable and safely parallel: an existing output is never recomputed, and each
volume is claimed with an atomic lock file before work starts, so any number of
workers can be pointed at the same list and they partition the work between
themselves. That matters here because the per-volume cost varies by a factor of
five across tasks -- an MSD liver series is 80-90 s and a lung series 18 s -- so
a static --shard split leaves workers idle while one grinds through the large
volumes. Dynamic claiming keeps every worker busy and lets capacity be added to
a run that is already going.

Usage
-----
    # start as many workers as the card and the CPU will carry; they coordinate
    python make_seg_cache.py --volumes matched_volumes.json --msd-root /path/MSD \
        --repo /path/BrokenEvidence-3D --device gpu:6 --fast
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--volumes", required=True,
                    help='JSON {organ: [volume_id, ...]}')
    ap.add_argument("--msd-root", required=True)
    ap.add_argument("--repo", required=True, help="BrokenEvidence-3D root")
    ap.add_argument("--device", default="gpu",
                    help='TotalSegmentator device string, e.g. gpu:6')
    ap.add_argument("--fast", action="store_true",
                    help="3 mm mode. Must match the mode the committed corpus "
                         "was built with -- see verify_seg_provenance.py, which "
                         "decides that empirically instead of assuming it.")
    ap.add_argument("--out-suffix", default="",
                    help="write to seg_cache{suffix}/ instead of seg_cache/, "
                         "for side-by-side comparison of modes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stale-minutes", type=float, default=20.0)
    ap.add_argument("--reverse", action="store_true",
                    help="walk the list from the end, so a second wave of "
                         "workers meets the first in the middle")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Pin before torch is imported, so the visible-device list the child sees
    # has exactly one card on it. TotalSegmentator's own "gpu:N" handling has
    # varied across versions; CUDA_VISIBLE_DEVICES does not.
    dev = args.device
    if dev.startswith("gpu:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
        dev = "gpu"

    from totalsegmentator.python_api import totalsegmentator

    todo: list[tuple[str, str, Path, Path]] = []
    volumes = json.load(open(args.volumes))
    for organ, vids in sorted(volumes.items()):
        cache = Path(args.repo) / f"cfqa_{organ}" / f"seg_cache{args.out_suffix}"
        cache.mkdir(parents=True, exist_ok=True)
        for vid in sorted(vids):
            vol = Path(args.msd_root) / organ / "imagesTr" / f"{vid}.nii.gz"
            out = cache / f"{vid}_seg.nii.gz"
            todo.append((organ, vid, vol, out))

    todo = [t for i, t in enumerate(sorted(todo, key=lambda t: (t[0], t[1])))
            if i % args.nshards == args.shard]
    if args.reverse:
        todo.reverse()
    if args.limit:
        todo = todo[: args.limit]

    def claim(lock: Path) -> bool:
        """Atomically claim a volume. False means another worker holds it.

        A lock whose holder died leaves the volume unsegmented forever, so a
        lock older than --stale-minutes with no output beside it is taken over.
        The window is generous: the slowest volumes here run 90 s, so anything
        past several minutes really is a dead worker and not a slow one.
        """
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = (time.time() - lock.stat().st_mtime) / 60.0
            except FileNotFoundError:
                return False
            if age > args.stale_minutes:
                print(f"stealing stale lock ({age:.1f} min) {lock.name}",
                      flush=True)
                lock.write_text(f"{os.getpid()} {time.time():.0f}\n")
                return True
            return False

    done = skipped = failed = held = 0
    t_start = time.time()
    for organ, vid, vol, out in todo:
        if out.exists() and out.stat().st_size > 0:
            skipped += 1
            continue
        if not vol.exists():
            print(f"MISSING VOLUME {vol}", flush=True)
            failed += 1
            continue
        lock = out.with_suffix(".lock")
        if not claim(lock):
            held += 1
            continue
        t0 = time.time()
        # PID in the temporary name as well: a stolen lock must not have two
        # processes writing the same bytes.
        tmp = out.with_suffix(f".part{os.getpid()}.nii.gz")
        try:
            # Write to a temporary name and rename: a killed process must not
            # leave a truncated mask that the resume logic then treats as done.
            totalsegmentator(str(vol), str(tmp), fast=args.fast, ml=True,
                             quiet=True, device=dev)
            tmp.rename(out)
            lock.unlink(missing_ok=True)
            done += 1
            print(f"[{done}] {organ}/{vid} {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:                     # keep going; report at end
            failed += 1
            tmp.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
            print(f"FAILED {organ}/{vid}: {exc!r}", flush=True)

    el = time.time() - t_start
    print(f"worker {os.getpid()}: {done} segmented, {skipped} already present, "
          f"{held} held by other workers, {failed} failed, {el / 60:.1f} min"
          + (f", {el / done:.1f}s per volume" if done else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
