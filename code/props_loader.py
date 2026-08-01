"""
props_loader.py  —  reads outputs.properties and returns F1-F8 outputs for a phase.

One helper, used by both the plumbing test and the real eval.
No repetition: the same loader serves 'before' and 'after' via the phase argument.
"""

FKEYS = [f"F{i}" for i in range(1, 9)]  # F1..F8

# Maps the short label used by the eval to the properties-file prefix
PHASE_PREFIX = {
    "before": "Before_fine_tuning_",
    "after":  "After_fine_tuning_",
}


def load_properties(path="outputs.properties"):
    """Parse a simple key = value properties file. Returns a dict."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # convert literal \n back into real newlines
            value = value.replace("\\n", "\n")
            data[key] = value
    return data


def get_outputs(phase, path="outputs.properties"):
    """
    Return {F1: output, F2: output, ... F8: output} for the given phase.
    phase = 'before' or 'after'.
    """
    if phase not in PHASE_PREFIX:
        raise ValueError(f"phase must be 'before' or 'after', got '{phase}'")
    prefix = PHASE_PREFIX[phase]
    data = load_properties(path)
    outputs = {}
    missing = []
    for fk in FKEYS:
        full_key = f"{prefix}{fk}"
        if full_key in data:
            outputs[fk] = data[full_key]
        else:
            missing.append(full_key)
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return outputs


def write_outputs(phase, outputs, path="outputs.properties"):
    """
    Write/replace the 8 outputs for a phase in the properties file.
    outputs = {'F1': text, ... 'F8': text}.
    Preserves other lines; replaces only this phase's keys.
    Used later by run_full_eval.py to auto-save real model outputs.
    """
    prefix = PHASE_PREFIX[phase]
    # read existing lines
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # build the new key=value lines for this phase (escape newlines)
    new_lines = {}
    for fk in FKEYS:
        val = outputs[fk].replace("\n", "\\n")
        new_lines[f"{prefix}{fk}"] = f"{prefix}{fk} = {val}\n"

    # replace existing keys in place; track which we still need to append
    remaining = set(new_lines.keys())
    out = []
    for line in lines:
        stripped = line.strip()
        key = stripped.split("=")[0].strip() if "=" in stripped else None
        if key in new_lines:
            out.append(new_lines[key])
            remaining.discard(key)
        else:
            out.append(line)

    # append any keys that weren't already present
    if remaining:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        for fk in FKEYS:
            k = f"{prefix}{fk}"
            if k in remaining:
                out.append(new_lines[k])

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
