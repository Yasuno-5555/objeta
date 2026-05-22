import sys, json

# Read all of stdin
raw = sys.stdin.read()
# Handle multiple JSON objects (there might be build warnings before the JSON)
# Find the first { and last }
start = raw.find('{')
end = raw.rfind('}')
if start >= 0 and end > start:
    data = json.loads(raw[start:end+1])
    skip = {'warm_runs'}
    out = {k: v for k, v in data.items() if k not in skip}
    print(json.dumps(out, indent=2))
else:
    print("No JSON found in output")
