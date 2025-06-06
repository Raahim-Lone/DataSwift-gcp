
#!/usr/bin/env python3
"""
parse_plan.py  –  canonical + robust (2025-05-19)
=================================================
Converts a Postgres EXPLAIN (FORMAT JSON) object into a PyG Data graph.

x[:,0] = operator-type ID   (canonical, deterministic)
x[:,1] = log1p(estimated rows)
x[:,2] = log1p(width)
x[:,3] = log1p(start-up cost)
x[:,4] = log1p(total  cost)
x[:,5] = fan-out (# direct child plans)
"""
from __future__ import annotations
import re, numpy as np, torch
from torch_geometric.data import Data
from modules.op_maps import NODE_TYPE_MAP, EDGE_TYPE_MAP

# ────────────────────────────── helpers ──────────────────────────────
_PAT_OP_FALLBACK = re.compile(r"(scan|join|agg|sort|hash|seek|filter|index)", re.I)
_WS_UNDERSCORE   = re.compile(r"[\s_]+")

def canonical(op: str) -> str:
    """lower-case and collapse ALL whitespace/underscores → single token."""
    return _WS_UNDERSCORE.sub("", op.lower())

def _unwrap_plan(root: dict) -> dict:
    """Return the dict that actually contains the top-level ‘Plan’."""
    if "Plan" in root and isinstance(root["Plan"], dict):
        return root["Plan"]
    if "plan" in root:                             # some tools use lower-case
        def dfs(obj):
            if isinstance(obj, dict):
                if "Plan" in obj:  return obj["Plan"]
                for v in obj.values():
                    res = dfs(v)
                    if res is not None:
                        return res
            elif isinstance(obj, list):
                for elt in obj:
                    res = dfs(elt)
                    if res is not None:
                        return res
            return None
        found = dfs(root["plan"])
        if found is not None:
            return found
    return root                                    # fallback (already a plan)

def _raw_op(node: dict) -> str:
    """Extract raw operator string from a Plan node."""
    for k, v in node.items():
        lk = k.lower()
        if "node" in lk and "type" in lk:          # normal path
            return str(v)
        if isinstance(v, str) and _PAT_OP_FALLBACK.search(v):
            return v                               # regex fallback
    return "Unknown"

def _find_num(node: dict, contains: list[str]) -> float:
    for k, v in node.items():
        if isinstance(v, (int, float)):
            lk = k.lower()
            if all(tok in lk for tok in contains):
                return float(v)
    return 0.0

def _fanout(node: dict) -> float:
    for v in node.values():
        if isinstance(v, list):
            return float(sum(1 for elt in v if isinstance(elt, dict)))
    return 0.0

# ─────────────────────────── feature builders ──────────────────────────
def _node_feats(node: dict) -> np.ndarray:
    op_can = canonical(_raw_op(node))
    NODE_TYPE_MAP.setdefault(op_can, len(NODE_TYPE_MAP))
    op_id  = NODE_TYPE_MAP[op_can]

    est   = np.log1p(_find_num(node, ["row"]))
    width = np.log1p(_find_num(node, ["width"]))
    scost = np.log1p(_find_num(node, ["startup", "cost"]))
    tcost = np.log1p(_find_num(node, ["total",   "cost"]))
    fan   = _fanout(node)

    return np.array([op_id, est, width, scost, tcost, fan], dtype=np.float32)

def _edge_type(child: dict) -> int:
    jt = child.get("Join Type")
    if jt is None:                       # try generic “… join …” key
        for k, v in child.items():
            if isinstance(v, str) and "join" in k.lower():
                jt = v
                break
    jt = jt or "child"
    jt_can = canonical(jt)
    EDGE_TYPE_MAP.setdefault(jt_can, len(EDGE_TYPE_MAP))
    return EDGE_TYPE_MAP[jt_can]

# ───────────────────────────── main API ────────────────────────────────
def parse_plan(raw_json: dict) -> Data:
    """
    Parameters
    ----------
    raw_json : dict
        The object you get from json.loads(<psql output>)
        – can be the whole EXPLAIN wrapper or already the “Plan” dict.

    Returns
    -------
    torch_geometric.data.Data
        .x  : (N, 6)  float32
        .edge_index : (2,E) long
        .edge_attr  : (E,)  long   (join-type IDs)
    """
    root      = _unwrap_plan(raw_json)
    feats, ei, ea = [], [], []

    def walk(node: dict, parent: int | None = None):
        idx = len(feats)
        feats.append(_node_feats(node))
        if parent is not None:
            ei.append((parent, idx))
            ea.append(_edge_type(node))
        for v in node.values():
            if isinstance(v, list):
                for elt in v:
                    if isinstance(elt, dict):
                        walk(elt, idx)

    walk(root)

    x  = torch.tensor(np.stack(feats), dtype=torch.float32)
    ei_t = torch.tensor(ei, dtype=torch.long).t() if ei else torch.empty((2,0), dtype=torch.long)
    ea_t = torch.tensor(ea, dtype=torch.long)     if ea else torch.empty((0,),  dtype=torch.long)
    return Data(x=x, edge_index=ei_t, edge_attr=ea_t)
