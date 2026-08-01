"""
test_plumbing.py  —  proves the automation machinery works, with NO model and NO DeepEval.

What it checks:
  1. The properties file can be read.
  2. get_outputs('before') returns all 8 F1-F8 outputs.
  3. get_outputs('after')  returns all 8 F1-F8 outputs.
  4. write_outputs() can overwrite a phase and read it back correctly (round-trip),
     including multi-line values with newlines.

If this passes, the file -> loader -> usable-data flow is sound, and we can safely
build the real Option B (run the model, write real outputs, feed the eval) on top of it.

Run:  python3 test_plumbing.py
"""
from props_loader import get_outputs, write_outputs, FKEYS

PASS = "PASS"
FAIL = "FAIL"

def check(label, condition):
    print(f"  [{PASS if condition else FAIL}] {label}")
    return condition

def main():
    all_ok = True

    print("=" * 60)
    print("PLUMBING TEST — properties file flow (no model, no DeepEval)")
    print("=" * 60)

    # 1 & 2 — read BEFORE phase
    print("\n1. Reading 'before' phase:")
    before = get_outputs("before")
    all_ok &= check("all 8 F-keys present", sorted(before.keys()) == FKEYS)
    all_ok &= check("F1 has content", len(before["F1"]) > 0)
    print(f"      sample  Before F1 = {before['F1'][:60]}...")

    # 3 — read AFTER phase
    print("\n2. Reading 'after' phase:")
    after = get_outputs("after")
    all_ok &= check("all 8 F-keys present", sorted(after.keys()) == FKEYS)
    all_ok &= check("F5 has content", len(after["F5"]) > 0)
    print(f"      sample  After  F5 = {after['F5'][:60]}...")

    # 4 — round-trip write with a multi-line value
    print("\n3. Round-trip write test (multi-line value):")
    multiline = "=== ENGINEERING INSIGHT ===\nTITLE: round-trip test\nSEVERITY: LOW\nLINE 3 here"
    sample = {fk: (multiline if fk == "F1" else f"roundtrip {fk}") for fk in FKEYS}
    write_outputs("before", sample, path="outputs.properties")
    reloaded = get_outputs("before")
    all_ok &= check("F1 multi-line preserved (has 3 newlines)", reloaded["F1"].count("\n") == 3)
    all_ok &= check("F1 content matches what we wrote", reloaded["F1"] == multiline)
    all_ok &= check("F8 content matches what we wrote", reloaded["F8"] == "roundtrip F8")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED — plumbing is sound.")
        print("Next: build run_full_eval.py to write REAL model outputs here.")
    else:
        print("RESULT: SOME CHECKS FAILED — fix before building Option B.")
    print("=" * 60)

    # restore dummy data note
    print("\nNote: this test overwrote the 'before' keys with round-trip test data.")
    print("That's expected — run_full_eval.py will replace them with real outputs.")

if __name__ == "__main__":
    main()
