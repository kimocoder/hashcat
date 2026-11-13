#!/usr/bin/env python3

##
## Author......: See docs/credits.txt
## License.....: MIT
##

# A/B benchmark of two hashcat builds. It runs `hashcat -b` for each mode against both binaries,
# repeats it, and reports the median and the relative change. Kernel compilation happens once per
# build and is excluded by the warmup run; the reported speed is the one hashcat prints for the
# kernel, so a slow first run only warms the cache.
#
# The default pair is this tree and the pristine checkout beside it:
#
#   ./kimobench.py
#
# The binary that does not exist yet is named in the error and the build line is printed. Any other
# pair works too:
#
#   ./kimobench.py --ours ./hashcat --theirs /path/to/other/hashcat
#   ./kimobench.py --modes 22000,2500 --runs 5 --device-types 1 --force
#
# A mode is benchmarked with the binary's own device selection unless --device-types is given, so
# pass -D 1 -f on a machine that only has the CPU backend.

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

DEFAULT_MODES = [22000, 22001, 2500, 2501, 16800, 16801]

SPEED_RE = re.compile(r"Speed\.#(\d+)\.+:\s+([0-9.]+)\s+([kMGT]?H/s)")
PARAM_RE = re.compile(r"@\s+Accel:(\S+)\s+Loops:(\S+)\s+Thr:(\S+)\s+Vec:(\S+)")

UNITS = {"H/s": 1.0, "kH/s": 1e3, "MH/s": 1e6, "GH/s": 1e9, "TH/s": 1e12}


def parse_speed(output):
  # One line per device. Keep the last value seen for each device so a repeated status line cannot
  # double count, then sum the devices for the machine's throughput.

  per_device = {}

  for dev, value, unit in SPEED_RE.findall(output):
    per_device[int(dev)] = float(value) * UNITS[unit]

  if len(per_device) == 0:
    return None

  params = PARAM_RE.search(output)

  return {
    "speed": sum(per_device.values()),
    "devices": len(per_device),
    "params": "/".join(params.groups()) if params else "",
  }


def run_once(binary, mode, args):
  cmd = [binary, "-b", "-m", str(mode), "--deprecated-check-disable"]

  if args.device_types is not None:
    cmd += ["-D", args.device_types]

  if args.force is True:
    cmd += ["--force"]

  if args.extra:
    cmd += args.extra.split()

  try:
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(binary)),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=args.timeout)
  except subprocess.TimeoutExpired:
    return None, "timeout after %ds" % args.timeout

  output = proc.stdout.decode(errors="replace")

  result = parse_speed(output)

  if result is None:
    reason = "no Speed line"

    for line in output.splitlines():
      if "Unknown hash-mode" in line or "No devices found" in line or "Kernel" in line and "error" in line.lower():
        reason = line.strip()
        break

    return None, reason

  return result, output


def benchmark(binary, mode, args):
  # warmup: compiles the kernels and lets autotune settle, not counted
  for _ in range(args.warmup):
    run_once(binary, mode, args)

  samples = []
  params = ""

  for _ in range(args.runs):
    result, detail = run_once(binary, mode, args)

    if result is None:
      return {"speed": None, "reason": detail, "samples": []}

    samples.append(result["speed"])
    params = result["params"]

  return {"speed": statistics.median(samples), "samples": samples, "params": params, "reason": ""}


def fmt_speed(speed):
  if speed is None:
    return "n/a"

  for unit, scale in (("TH/s", 1e12), ("GH/s", 1e9), ("MH/s", 1e6), ("kH/s", 1e3)):
    if speed >= scale:
      return "%.2f %s" % (speed / scale, unit)

  return "%.0f H/s" % speed


def main():
  ap = argparse.ArgumentParser(description="A/B benchmark of two hashcat builds")

  here = os.path.dirname(os.path.abspath(__file__))

  ap.add_argument("--ours", default=os.path.join(here, "hashcat"), help="binary under test")
  ap.add_argument("--theirs", default=os.path.join(os.path.dirname(here), "hashcat-upstream", "hashcat"),
                  help="reference binary")
  ap.add_argument("--modes", default=",".join(str(m) for m in DEFAULT_MODES),
                  help="comma separated hash modes")
  ap.add_argument("--runs", type=int, default=3, help="measured runs per binary and mode")
  ap.add_argument("--warmup", type=int, default=1, help="uncounted runs before measuring")
  ap.add_argument("-D", dest="device_types", default=None, help="pass -D to hashcat")
  ap.add_argument("-f", dest="force", action="store_true", help="pass --force to hashcat")
  ap.add_argument("--timeout", type=int, default=600, help="seconds per hashcat invocation")
  ap.add_argument("--extra", default="", help="extra arguments, quoted")
  ap.add_argument("--json", default=None, help="write the result as JSON to this path")

  args = ap.parse_args()

  modes = [int(m) for m in args.modes.split(",") if m.strip()]

  missing = False

  for binary in (args.ours, args.theirs):
    if os.path.isfile(binary) is not True:
      print("%s not found" % binary)
      print("build it first, for example: make -C %s -j4" % os.path.dirname(os.path.abspath(binary)))
      missing = True

  if missing is True:
    return 1

  if args.runs < 1:
    return 1

  print("ours   %s" % os.path.abspath(args.ours))
  print("theirs %s" % os.path.abspath(args.theirs))
  print()

  rows = []

  header = "%-8s %14s %14s %10s %-22s" % ("mode", "ours", "theirs", "change", "accel/loops/thr/vec")

  print(header)
  print("-" * len(header))

  failed = False

  for mode in modes:
    ours = benchmark(os.path.abspath(args.ours), mode, args)
    theirs = benchmark(os.path.abspath(args.theirs), mode, args)

    if ours["speed"] is None or theirs["speed"] is None:
      failed = True

    if ours["speed"] is not None and theirs["speed"] is not None:
      change = (ours["speed"] - theirs["speed"]) / theirs["speed"] * 100.0
      change_str = "%+.1f %%" % change
    else:
      change_str = "n/a"

    params = ours["params"] or theirs["params"]

    print("%-8d %14s %14s %10s %-22s" % (mode, fmt_speed(ours["speed"]), fmt_speed(theirs["speed"]), change_str, params))

    if ours["speed"] is None:
      print("         ours:   %s" % ours["reason"])

    if theirs["speed"] is None:
      print("         theirs: %s" % theirs["reason"])

    rows.append({
      "mode": mode,
      "ours": ours["speed"],
      "theirs": theirs["speed"],
      "change_pct": (None if (ours["speed"] is None or theirs["speed"] is None)
                     else (ours["speed"] - theirs["speed"]) / theirs["speed"] * 100.0),
      "ours_samples": ours["samples"],
      "theirs_samples": theirs["samples"],
      "params": params,
    })

  print()

  if args.json:
    with open(args.json, "w") as f:
      json.dump({"ours": os.path.abspath(args.ours), "theirs": os.path.abspath(args.theirs),
                 "rows": rows}, f, indent=2)

    print("wrote %s" % args.json)

  return 1 if failed is True else 0


if __name__ == "__main__":
  sys.exit(main())