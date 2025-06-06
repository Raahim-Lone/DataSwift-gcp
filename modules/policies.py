#!/usr/bin/env python3
"""
policies.py – optimisation policies for the IMDB benchmark.

make_policy(name[, tau]) → callable(sql) → hint_combo
    name ∈ {'default', 'imc', 'eim', 'full'}
    tau  : similarity threshold for EIM (only for 'eim' and 'full')
"""

from __future__ import annotations
import os, re, json, hashlib, sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch, joblib, psycopg2
from sentence_transformers import SentenceTransformer
from torch_geometric.data.data import DataEdgeAttr
from getpass import getuser
import torch.serialization as serialization

# ─── local imports ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.eim         import EmbeddingMemory
from modules.bandit      import ThompsonBandit
from modules.parse_plan  import parse_plan, canonical as _can
from modules.plan2vec    import Plan2VecEncoder
from modules.train_gnimc import GNIMCModel

# ─── artefacts & static data ──────────────────────────────────────────
MODELS_DIR = Path.home() / "models"
DATA_ROOT  = Path.home() / "New_Data"

PIPELINE   = joblib.load(MODELS_DIR / "pipeline.pkl")
HINT_IDS   = json.load(open(MODELS_DIR / "hint_ids.json"))
SQL_VOCAB  = json.load(open(MODELS_DIR / "sql_vocab.json"))
SQL2IDX    = json.load(open(MODELS_DIR / "sql2idx.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_train = torch.from_numpy(
    PIPELINE.transform(
        np.load("/Users/raahimlone/rahhh/new/embeddings/syntaxA_embedding.npy")
    )
).float().to(DEVICE)

serialization.add_safe_globals([DataEdgeAttr])

# ─── build encoder + GN-IMC head once ─────────────────────────────────
def _load_models():
    numeric_dim = (
        parse_plan(
            json.load(open(next((Path.home() / "Downloads" / "dsb").rglob("*.json"))))
        ).x.size(1) - 1
    )
    enc = Plan2VecEncoder(
        num_op_types=len(json.load(open(MODELS_DIR / "op_map.json"))),
        numeric_dim=numeric_dim,
        vocab_size=len(SQL_VOCAB),
        hidden_dim=256,
        num_layers=3,
        out_dim=512,
    ).to(DEVICE)

    y_mat = np.load(DATA_ROOT / "Y_scalednew.npz")["Y"]
    head = GNIMCModel(
        q_dim=enc.out_dim + 120,
        h_dim=y_mat.shape[1],
        rank=128,
        num_hints=len(HINT_IDS),
    ).to(DEVICE)

    enc_ck = torch.load(MODELS_DIR / "plan2vec_ckpt.pt", map_location=DEVICE)
    if enc_ck["edge_embed.weight"].size(0) != enc.edge_embed.weight.size(0):
        enc.edge_embed = torch.nn.Embedding(
            enc_ck["edge_embed.weight"].size(0),
            enc.edge_embed.weight.size(1),
            padding_idx=0,
        ).to(DEVICE)

    enc.load_state_dict(enc_ck, strict=False)
    head.load_state_dict(
        torch.load(MODELS_DIR / "gnimc_ckpt.pt", map_location=DEVICE)["model_state_dict"]
    )
    enc.eval(); head.eval()
    return enc, head, torch.from_numpy(y_mat).float().to(DEVICE)

ENC, HEAD, H_ALL = _load_models()

# mean / std of log-latencies (for de-/re-scaling)
def _mu_sig():
    W = np.load(DATA_ROOT / "Wnew.npy")
    M = np.load(DATA_ROOT / "Mnew.npy")
    return np.log1p(W[M > 0]).mean(), np.log1p(W[M > 0]).std()

MU_Y, SIG_Y = _mu_sig()

# ─── helpers ──────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
def _plan_cost(cur, sql: str, hint: str) -> float:
    toggles = "" if not hint else _toggle(hint)
    cur.execute(f"{toggles} EXPLAIN (FORMAT JSON) {sql}")
    obj = cur.fetchone()[0]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, list):
        obj = obj[0]
    return float(obj["Plan"]["Total Cost"])

def _pca_vec(sql: str) -> torch.Tensor:
    can = _can(sql)
    if can in SQL2IDX:
        return embed_train[SQL2IDX[can]:SQL2IDX[can]+1]
    vec = embedder.encode([sql], normalize_embeddings=True)
    return torch.from_numpy(PIPELINE.transform(vec)).float().to(DEVICE)

def _sql_tokens(sql: str, maxlen: int = 128):
    toks  = re.findall(r"\w+", sql.lower())[:maxlen]
    ids   = [SQL_VOCAB.get(t, 1) for t in toks] + [0]*(maxlen-len(toks))
    mask  = [1]*len(toks) + [0]*(maxlen-len(toks))
    return (torch.tensor([ids], dtype=torch.long, device=DEVICE),
            torch.tensor([mask], dtype=torch.float32, device=DEVICE))

def _encode(sql: str) -> torch.Tensor:
    conn = psycopg2.connect(
        dbname="IMDB", user=getuser(), host="localhost",
        port="6543", password=os.getenv("PGPASSWORD", "")
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    obj = cur.fetchone()[0]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, list):
        obj = obj[0]
    plan = parse_plan(obj).to(DEVICE)
    plan.batch = torch.zeros(plan.x.size(0), dtype=torch.long, device=DEVICE)
    cur.close(); conn.close()

    ids, mask = _sql_tokens(sql)
    with torch.no_grad():
        z_core = ENC(plan, ids, mask)
    return torch.cat([z_core, _pca_vec(sql)], dim=1)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

# ─── GLOBALS for v3 bandit logic ──────────────────────────────────────
best_hint: Dict[str, Tuple[int, float]] = {}   # qkey → (hint_idx, latency_ms)
K_TOP = 5                                      # GN-IMC top-K to expose as arms

# ─── policy factory ───────────────────────────────────────────────────
def make_policy(name: str, tau: float | None = None) -> Callable[[str], str]:
    """
    name : 'default' | 'imc' | 'eim' | 'full'
    tau  : similarity threshold for EIM in {'eim','full'}
    """
    name = name.lower()
    if name not in {"default", "imc", "eim", "full"}:
        raise ValueError(f"unknown policy {name!r}")

    # — 1. DEFAULT ————————————————————————————————————————————————
    if name == "default":
        def _default(sql: str) -> str:
            return ""
        return _default

    # — 2. IMC-only ————————————————————————————————————————————————
    if name == "imc":
        def _imc(sql: str) -> str:
            z = _encode(sql)
            with torch.no_grad():
                mu, sig, _ = HEAD(z, H_ALL)
            mu, sig = mu.cpu().numpy(), sig.cpu().numpy()
            score   = mu + 0.7 * sig
            idx     = int(score.argmin())
            hint    = HINT_IDS[idx]

            # cost-guard
            conn = psycopg2.connect(dbname="IMDB", user=getuser(),
                                    host="localhost", port="6543",
                                    password=os.getenv("PGPASSWORD",""))
            conn.autocommit = True; cur  = conn.cursor()
            try:
                if _plan_cost(cur, sql, hint) > 1.5 * _plan_cost(cur, sql, ""):
                    hint = ""
            finally:
                cur.close(); conn.close()
            return hint
        return _imc

    # — 3 & 4.  EIM (+ bandit) policies ————————————————
    tau = 0.15 if tau is None else tau
    eim    = EmbeddingMemory(dim=ENC.out_dim + 120, tau=tau)
    bandit = ThompsonBandit() if name == "full" else None

    def _pick(sql: str, observed_ms: float | None = None) -> str:
        qkey = _hash(sql)
        z    = _encode(sql)

        # —— GN-IMC predictions (µ, σ, latencies) ——
        with torch.no_grad():
            mu, sig, _ = HEAD(z, H_ALL)
        mu_np, sig_np = mu.cpu().numpy(), sig.cpu().numpy()
        lat_mu = np.expm1((mu_np * SIG_Y) + MU_Y) * 1e3

        # —— Candidate arms (v3) ————————————————————
        arms: Dict[str, Tuple[int | None, float]] = {
            "default": (None, float(lat_mu.min()))
        }

        # (A) cache neighbour
        nb_idx, nb_lat = eim.query(z.cpu().numpy())
        if nb_idx is not None:
            arms["cache"] = (nb_idx, nb_lat)

        # (B) best of k-NN (k=20) if <60 s
        kn_idx, kn_lat = eim.best_of_knn(z.cpu().numpy(), k=20)
        if kn_idx is not None and kn_lat < 60_000:
            arms["knn"] = (kn_idx, kn_lat)

        # (C) IMC top-K (µ + 0.5 σ penalty)
        for rank, idx in enumerate(np.argsort(lat_mu)[:K_TOP], 1):
            penalty = 1 + 0.5 * sig_np[idx]
            arms[f"imc{rank}"] = (int(idx), float(lat_mu[idx] * penalty))

        # (D) best real hint so far
        if qkey in best_hint:
            arms["best"] = best_hint[qkey]

        # —— Bandit choice ————————————————————————
        if bandit is None:          # happens for 'eim' policy
            # deterministic fallback: cache → imc1
            choice = "cache" if "cache" in arms else "imc1"
        else:
            choice = bandit.choose(qkey, arms)

        # final hint combo
        hint = "" if choice == "default" else HINT_IDS[arms[choice][0]]

        # —— cost guard (one last safety net) ——————————
        if hint:
            conn = psycopg2.connect(dbname="IMDB", user=getuser(),
                                    host="localhost", port="6543",
                                    password=os.getenv("PGPASSWORD",""))
            conn.autocommit = True; cur = conn.cursor()
            try:
                if _plan_cost(cur, sql, hint) > 1.5 * _plan_cost(cur, sql, ""):
                    hint, choice = "", "default"
            finally:
                cur.close(); conn.close()

        # —— feedback & memory updates —————————————
        if observed_ms is not None:
            # bandit feedback
            if bandit is not None:
                bandit.update(qkey, choice, observed_ms)
            # remember best real hint
            if hint:
                idx = HINT_IDS.index(hint)
                if (qkey not in best_hint) or (observed_ms < best_hint[qkey][1]):
                    best_hint[qkey] = (idx, observed_ms)
                eim.add(z.cpu().numpy(), idx, observed_ms)

        return hint

    # adapter for evaluate_imdb.py
    def _policy(sql: str) -> str:
        return _pick(sql)

    def _feedback(sql: str, ms: float):
        _pick(sql, observed_ms=ms)

    _policy.feedback = _feedback
    return _policy
