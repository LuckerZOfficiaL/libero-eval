import json, sys
d = json.loads(sys.stdin.read())
for k, v in d.items():
    if isinstance(v, dict) and "lib" in v:
        m = lambda x: 100 * sum(y for y in x.values() if y is not None) / max(1, sum(y is not None for y in x.values()))
        print(k, "raw lib %.1f new %.1f" % (m(v["lib"]), m(v["new"])))
