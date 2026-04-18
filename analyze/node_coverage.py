"""
Analyze node_trace.log to find which nodes were called and how often.
Compares against nodes defined in src/agent/nodes/*.py that emit [node] logs.
"""
import re
from collections import Counter
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "logs" / "node_trace.log"
NODES_DIR = Path(__file__).parent.parent / "src" / "agent" / "nodes"

# Extract all [node] names from source files
def get_defined_nodes():
    nodes = {}
    for f in NODES_DIR.glob("*.py"):
        matches = re.findall(r'\[node\]\s+([\w_]+)', f.read_text(encoding="utf-8"))
        for m in matches:
            nodes[m] = f.name
    return nodes

# Count [node] calls in log
def get_log_counts():
    text = LOG_FILE.read_text(encoding="utf-8")
    matches = re.findall(r'\[node\]\s+([\w_]+)', text)
    return Counter(matches)

def main():
    defined = get_defined_nodes()
    counts = get_log_counts()

    total = sum(counts.values())
    print(f"Total [node] calls in log: {total}\n")

    print(f"{'Node':<25} {'File':<20} {'Calls':>6}  {'% of total':>10}")
    print("-" * 65)
    for node, src_file in sorted(defined.items()):
        c = counts.get(node, 0)
        pct = f"{c/total*100:.1f}%" if total else "N/A"
        flag = "  <-- NEVER CALLED" if c == 0 else ("  <-- RARE" if c < 5 else "")
        print(f"{node:<25} {src_file:<20} {c:>6}  {pct:>10}{flag}")

    unknown = set(counts) - set(defined)
    if unknown:
        print(f"\nIn log but not in source: {unknown}")

if __name__ == "__main__":
    main()
