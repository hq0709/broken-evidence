"""
VRAM-aware job queue for a fixed set of GPUs.

Why not `spatialgen/gpulock.sh`: that script assumes a whole idle card per job.
Here the two cards available are not equivalent -- one is exclusively ours, the
other is shared with another user's long-running job -- so the scheduler has to
respect a per-card free-memory budget rather than a per-card job count.
`MontageModel` scores each option with one unbatched forward pass, so a single
process never saturates an H100's compute; throughput comes from packing as
many processes onto a card as its free memory allows.

Job format (JSONL, one object per line):
    {"label": "id_Task03_Liver_qwen32b_plain",
     "out":   "results_new/id_Task03_Liver_qwen32b_plain.jsonl",
     "vram_gb": 64,
     "gpu": 6,                       # optional: pin. omitted = any card that fits
     "cmd": ["python", "spatialgen/run_identification_control.py", ...]}

Guarantees that matter for a queue left running unattended:
  * a job writes to `<out>.part` and the runner renames it only on exit 0, so a
    killed job never leaves a short file that a later run mistakes for a
    finished one;
  * `--skip-existing` skips only jobs whose final output is already there;
  * a job is claimed with an atomic lock file before it starts, so one queue per
    card can be pointed at the SAME job list and they partition it between
    themselves. That is what lets a card that has drained its share pick up the
    slow model's remaining runs instead of going idle;
  * a job's stdout/stderr goes to logs/<label>.log, never to the shared console,
    so a failure is diagnosable after the fact;
  * the budget is read from the card at start-up (nvidia-smi), so the memory
    another user's process holds is subtracted rather than assumed away.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESERVE_GB = 4.0        # headroom per card for activations, fragmentation, cuBLAS


def free_gb(gpu: int) -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=True).stdout.strip()
    total, used = (float(x) for x in out.split(","))
    return (total - used) / 1024.0


def foreign_gb(gpu: int) -> tuple[float, set[str]]:
    """Memory on `gpu` held by users other than this one, and who they are."""
    me = os.environ.get("USER") or str(os.getuid())
    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    total, who = 0.0, set()
    for line in apps:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] != uuid:
            continue
        owner = subprocess.run(["ps", "-o", "user=", "-p", parts[1]],
                               capture_output=True, text=True).stdout.strip()
        if owner and owner != me:
            total += float(parts[2]) / 1024.0
            who.add(owner)
    return total, who


def share_aware_budget(gpu: int, headroom: float) -> float:
    """Budget that leaves a co-tenant room to grow.

    Scheduling against free memory is right on a card we own and wrong on a
    shared one: a neighbour's training job that has allocated 10 GB so far may
    still climb to its steady-state peak, and taking the rest of the card means
    it dies rather than us. `headroom` is what we promise to leave them beyond
    what they already hold -- set it from what their job has peaked at before,
    not from what it holds right now.
    """
    total = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True).stdout.strip()
    cap = float(total) / 1024.0
    held, who = foreign_gb(gpu)
    if not who:
        return max(0.0, cap - RESERVE_GB)
    budget = max(0.0, cap - held - headroom)
    print(f"gpu{gpu} is shared with {', '.join(sorted(who))} "
          f"({held:.1f} GB held); leaving them {headroom:.0f} GB of headroom, "
          f"taking {budget:.1f} GB", flush=True)
    return budget


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="6,7")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--pause-file", default="",
                    help="while this file exists, start no new jobs. Lets a run "
                         "that needs BOTH cards to itself -- the 72B sharded "
                         "across them -- take them without killing this queue "
                         "and losing its place in the job list.")
    ap.add_argument("--share-headroom", type=float, default=0.0,
                    help="GB to leave free for each co-tenant on a shared card. "
                         "When set, the budget is computed from what other users "
                         "already hold plus this, instead of from free memory.")
    ap.add_argument("--budget", default="",
                    help='explicit per-card budget in GB, e.g. "6:76,7:24". '
                         "Use this when a card is shared with someone else's "
                         "job: scheduling against measured free memory would "
                         "hand us every byte their allocator has not touched "
                         "yet, and taking it can OOM their run.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    logdir = repo / args.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.share_headroom:
        budget = {g: share_aware_budget(g, args.share_headroom) for g in gpus}
    else:
        budget = {g: max(0.0, free_gb(g) - RESERVE_GB) for g in gpus}
    for item in filter(None, args.budget.split(",")):
        g, gb = item.split(":")
        budget[int(g)] = float(gb)
    print("free VRAM after reserve: "
          + ", ".join(f"gpu{g} {b:.1f} GB" for g, b in budget.items()), flush=True)

    jobs = [json.loads(l) for l in open(args.jobs) if l.strip()]
    queue = []
    for j in jobs:
        out = repo / j["out"]
        if args.skip_existing and out.exists() and out.stat().st_size > 0:
            print(f"skip (done): {j['label']}", flush=True)
            continue
        want = float(j["vram_gb"])
        cands = [j["gpu"]] if "gpu" in j else gpus
        if not any(want <= budget[g] for g in cands if g in budget):
            print(f"IMPOSSIBLE: {j['label']} wants {want:.0f} GB; "
                  f"largest budget is "
                  f"{max(budget[g] for g in cands if g in budget):.1f} GB",
                  flush=True)
            continue
        queue.append(j)

    # Largest first: a 64 GB job queued behind two 16 GB ones can never start on
    # a card the small jobs have already filled.
    queue.sort(key=lambda j: -float(j["vram_gb"]))
    print(f"{len(queue)} jobs queued", flush=True)
    if args.dry_run:
        for j in queue:
            print(f"  [{j.get('gpu', 'any')}] {float(j['vram_gb']):5.1f} GB  "
                  f"{j['label']}")
        return

    running: list[dict] = []
    results: list[dict] = []
    t0 = time.time()
    while queue or running:
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            budget[r["gpu"]] += r["vram"]
            j = r["job"]
            part, final = repo / (j["out"] + ".part"), repo / j["out"]
            ok = rc == 0 and part.exists() and part.stat().st_size > 0
            if ok:
                part.replace(final)
            # Release the claim either way: a success is recorded by its output
            # file, and a failure must stay retryable rather than look claimed
            # to every future queue.
            (logdir / f"{j['label']}.claim").unlink(missing_ok=True)
            el = (time.time() - r["t0"]) / 60.0
            print(f"[{'ok ' if ok else 'FAIL'}] {j['label']} "
                  f"gpu{r['gpu']} rc={rc} {el:.1f} min"
                  + ("" if ok else f"  -> {logdir / (j['label'] + '.log')}"),
                  flush=True)
            results.append({"label": j["label"], "gpu": r["gpu"], "rc": rc,
                            "ok": ok, "minutes": round(el, 2)})

        if args.pause_file and os.path.exists(args.pause_file) and not running:
            time.sleep(args.poll)
            continue

        started = True
        while started and queue:
            started = False
            if args.pause_file and os.path.exists(args.pause_file):
                break
            for j in list(queue):
                want = float(j["vram_gb"])
                cands = [j["gpu"]] if "gpu" in j else sorted(
                    gpus, key=lambda g: -budget[g])
                for g in cands:
                    if g in budget and want <= budget[g]:
                        queue.remove(j)
                        # Re-check the output at claim time, not only at start-up.
                        # The claim is released when a job finishes so failures
                        # stay retryable, which means a queue on the other card
                        # can claim a job the first card has since COMPLETED and
                        # run it again.
                        if args.skip_existing:
                            fin = repo / j["out"]
                            if fin.exists() and fin.stat().st_size > 0:
                                print(f"skip (done elsewhere): {j['label']}",
                                      flush=True)
                                started = True
                                break
                        # Claim it. Another queue on another card may be reading
                        # the same job list; whoever creates the lock owns the
                        # job, and the loser simply moves on.
                        lock = logdir / f"{j['label']}.claim"
                        try:
                            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                            os.write(fd, f"{os.getpid()} gpu{g}\n".encode())
                            os.close(fd)
                        except FileExistsError:
                            print(f"claimed elsewhere, skipping: {j['label']}",
                                  flush=True)
                            started = True          # keep filling this card
                            break
                        budget[g] -= want
                        env = dict(os.environ)
                        env["CUDA_VISIBLE_DEVICES"] = str(g)
                        env["VOLUMETRIC_ROOT"] = str(repo)
                        # every job is written against cuda:0 of its own
                        # single-card view, so the pin above is the only place
                        # a device number appears
                        cmd = [c.replace("{OUT}", j["out"] + ".part")
                               for c in j["cmd"]]
                        log = open(logdir / f"{j['label']}.log", "w")
                        p = subprocess.Popen(cmd, cwd=repo, env=env,
                                             stdout=log, stderr=subprocess.STDOUT)
                        running.append({"job": j, "proc": p, "gpu": g,
                                        "vram": want, "t0": time.time()})
                        print(f"[start] {j['label']} gpu{g} ({want:.0f} GB, "
                              f"{budget[g]:.1f} GB left)", flush=True)
                        started = True
                        break
                if started:
                    break
        if running:
            time.sleep(args.poll)

    ok = sum(r["ok"] for r in results)
    print(f"\n{ok}/{len(results)} jobs ok in {(time.time() - t0) / 60:.1f} min")
    for r in results:
        if not r["ok"]:
            print(f"  FAILED {r['label']} (rc={r['rc']})")
    json.dump(results, open(logdir / "queue_summary.json", "w"), indent=1)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
