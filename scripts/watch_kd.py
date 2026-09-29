"""Restart a distillation trainer after the process has exited.

A quiet log is not a hang. This process does not kill a running trainer.
A graphics-card fault ends the trainer; this starts one new trainer with
--resume from the last checkpoint. Five deaths inside three minutes stop
the loop and write WATCH_STOP.

Default paths are the English 3k round.
"""
import argparse
import os
import subprocess
import time

ROOT = os.path.expanduser("~/sentinel-laya")
PY = os.path.join(ROOT, ".venv-stable/bin/python")
QUICK_S = 180
QUICK_LIMIT = 5

LOG = ""
WATCH_LOG = ""
INFLIGHT = ""
TRAINER_PID = ""
OUTPUT_FLAG = ""
ITEMS = ""
HOLDOUT = ""
OUTPUT = ""
MODEL_ID = ""
STATE_DIR = ""


def _note(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg + "\n"
    os.makedirs(os.path.dirname(WATCH_LOG), exist_ok=True)
    with open(WATCH_LOG, "a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    with open(LOG, "a") as handle:
        handle.write("WATCH " + line)
        handle.flush()
        os.fsync(handle.fileno())


def _tail(n=12000):
    if not os.path.exists(LOG):
        return ""
    with open(LOG, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - n))
        return handle.read().decode("utf-8", "replace")


def _train_ok():
    return "\nTRAIN_OK" in _tail(2_000_000) or _tail(2_000_000).startswith("TRAIN_OK")


def _trainer_pids():
    found = []
    me = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        if "python/train_kd.py" in cmd and OUTPUT_FLAG in cmd:
            found.append(pid)
    return found


def _start():
    if os.path.exists(INFLIGHT):
        os.remove(INFLIGHT)
        _note("removed inflight file before start")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["KD_FULL_ATTN"] = "1"
    env["PYTHONPATH"] = os.path.join(ROOT, "python")
    log_f = open(LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        [
            PY, "-u", "python/train_kd.py",
            "--model-id", MODEL_ID,
            "--items", ITEMS,
            "--holdout", HOLDOUT,
            "--output", OUTPUT,
            "--state-dir", STATE_DIR,
            "--epochs", "2",
            "--save-steps", "100",
            "--precision", "bf16",
            "--keep-temperature",
            "--resume",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    with open(TRAINER_PID, "w") as handle:
        handle.write(str(proc.pid) + "\n")
    _note(f"started trainer pid {proc.pid}")
    return proc, log_f


def main():
    global LOG, WATCH_LOG, INFLIGHT, TRAINER_PID, OUTPUT_FLAG, ITEMS, HOLDOUT, OUTPUT, MODEL_ID, STATE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/en3k_train.log")
    ap.add_argument("--state-dir", default="results/en3k")
    ap.add_argument("--items", default="data/en3k_items.pt")
    ap.add_argument("--holdout", default="data/en3k_holdout.pt")
    ap.add_argument("--output", default="models/laya-distill-en3k")
    ap.add_argument("--model-id", default="models/laya-distill-extra")
    args = ap.parse_args()
    LOG = os.path.join(ROOT, args.log) if not os.path.isabs(args.log) else args.log
    STATE_DIR = args.state_dir
    WATCH_LOG = os.path.join(ROOT, args.state_dir, "watch.log")
    INFLIGHT = os.path.join(ROOT, args.state_dir, "kd_inflight.txt")
    TRAINER_PID = os.path.join(ROOT, args.state_dir, "trainer.pid")
    OUTPUT = args.output
    OUTPUT_FLAG = f"--output {args.output}"
    ITEMS = args.items
    HOLDOUT = args.holdout
    MODEL_ID = args.model_id

    os.chdir(ROOT)
    if not os.path.isfile(os.path.join(ROOT, ITEMS)):
        _note(f"WATCH_STOP missing {ITEMS}")
        return
    quick = 0
    _note(f"watcher up pid {os.getpid()}")
    while True:
        if _train_ok():
            _note("training finished, watcher exit")
            return
        running = _trainer_pids()
        if running:
            _note(f"trainer already running {running}")
            while _trainer_pids():
                time.sleep(20)
            continue
        if quick >= QUICK_LIMIT:
            _note("WATCH_STOP five deaths under three minutes, not restarting")
            return
        proc, log_f = _start()
        started = time.time()
        code = proc.wait()
        log_f.close()
        lived = time.time() - started
        _note(f"trainer pid {proc.pid} exited {code} after {lived:.0f}s")
        if _train_ok():
            _note("training finished, watcher exit")
            return
        tail = _tail()
        if "FATAL_RESUME" in tail or "REFUSE_EXPORT" in tail:
            _note("WATCH_STOP bad checkpoint or export, not restarting")
            return
        if lived < QUICK_S:
            quick += 1
        else:
            quick = 0
        time.sleep(15)


if __name__ == "__main__":
    main()
