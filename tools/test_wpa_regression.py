#!/usr/bin/env python3

##
## Author......: See docs/credits.txt
## License.....: MIT
##

# Regression checks for the WPA binary hashfile path.
#
# A hccapx record is arbitrary bytes and may contain 0x0a, which a line reader would split. The
# loader has to sniff the file and hand it to the binary parser, so the first check builds a file
# where such a byte is guaranteed to be present, and expects both records to crack in 22000 and
# 2500. The second check pins the one deliberate difference between the 2500 and 2501 decoders: a
# record whose EAPOL key version is 0 is refused by 2500 and accepted by 2501.
#
# The records are built with tools/test_modules/m02500.py, so no fixture file is stored in the tree.

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile

TDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TDIR)
BIN  = os.path.join(ROOT, "hashcat")

sys.path.insert(0, os.path.join(TDIR, "test_modules"))

import m02500

WORD  = b"hashcat!"
ESSID = "hashcat-essid"
BSSID = bytes.fromhex("001122334455")
STMAC = bytes.fromhex("66778899aabb")


def make_record(seed, keyver=2, variant=0):
  snonce = bytes([0x33 + variant]) * 32
  anonce = bytes([seed]) + bytes([(0x44 + variant) & 0xff]) * 31

  eapol = m02500._gen_random_wpa_eapol(keyver, snonce)

  b64 = m02500.module_generate_hash(WORD, ESSID, None, BSSID, STMAC, snonce, anonce, keyver, eapol)

  return base64.b64decode(b64)


def make_record_with_newline(variant):
  for seed in range(256):
    rec = make_record(seed, 2, variant)

    if b"\x0a" in rec:
      return rec

  raise SystemExit("could not build a record containing 0x0a")


def set_keyver_zero(rec):
  buf = bytearray(rec)

  buf[42] = 0

  # the decoder prefers the EAPOL frame's key information word over the record's own byte, and that
  # word is big endian; the key version is its low two bits

  key_info = (buf[142] << 8) | buf[143]

  key_info &= ~3

  buf[142] = key_info >> 8
  buf[143] = key_info & 0xff

  return bytes(buf)


def run(mode, hashfile, wordlist, device_types, force):
  cmd = [BIN, "-m", str(mode), hashfile, wordlist, "--potfile-disable", "--logfile-disable", "--deprecated-check-disable"]

  if device_types is not None:
    cmd += ["-D", device_types]

  if force is True:
    cmd += ["--force"]

  proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  return proc.returncode, proc.stdout.decode(errors="replace")


def check(name, output, expected):
  if expected in output:
    print("ok   %s" % name)

    return True

  print("FAIL %s: expected %r" % (name, expected))
  print(output)

  return False


def main():
  ap = argparse.ArgumentParser(description="regression checks for the WPA hccapx path")

  ap.add_argument("-D", dest="device_types", default=None, help="pass -D to hashcat")
  ap.add_argument("-f", dest="force", action="store_true", help="pass --force to hashcat")
  ap.add_argument("--keep", action="store_true", help="keep the temporary directory")

  args = ap.parse_args()

  if os.path.isfile(BIN) is False:
    raise SystemExit("%s not found, build the tree first" % BIN)

  tmp = tempfile.mkdtemp(prefix="hashcat-wpa-")

  try:
    wordlist = os.path.join(tmp, "words.txt")

    with open(wordlist, "w") as f:
      f.write("wrongpass\n")
      f.write(WORD.decode() + "\n")

    rec_a = make_record_with_newline(0)
    rec_b = make_record_with_newline(1)

    two = os.path.join(tmp, "two.hccapx")

    with open(two, "wb") as f:
      f.write(rec_a)
      f.write(rec_b)

    ok = True

    code, out = run(22000, two, wordlist, args.device_types, args.force)
    ok = check("two records with 0x0a crack in 22000", out, "Recovered........: 2/2") and ok

    code, out = run(2500, two, wordlist, args.device_types, args.force)
    ok = check("two records with 0x0a crack in 2500", out, "Recovered........: 2/2") and ok

    kv0 = os.path.join(tmp, "keyver0.hccapx")

    with open(kv0, "wb") as f:
      f.write(set_keyver_zero(rec_a))

    code, out = run(2500, kv0, wordlist, args.device_types, args.force)
    ok = check("key version 0 is refused by 2500", out, "No hashes loaded.") and ok

    code, out = run(2501, kv0, wordlist, args.device_types, args.force)
    ok = check("key version 0 is loaded by 2501", out, "Recovered........: 0/1") and ok

    return 0 if ok is True else 1
  finally:
    if args.keep is True:
      print("kept %s" % tmp)
    else:
      shutil.rmtree(tmp)


if __name__ == "__main__":
  sys.exit(main())