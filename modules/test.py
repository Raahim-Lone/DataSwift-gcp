#!/usr/bin/env python3
"""
test.py ­– Batch Hint Recommender v4-b (2025-05-24)
────────────────────────────────────────────────────
* Adaptive Embedding-Indexed Memory v2
* Model-Aware Thompson Bandit
* Guaranteed no-regression baseline
* Command-line flags:
      --sql-dir   directory with *.sql templates   (required)
      --out       directory to drop a run log     (default: ./results)
"""
# ─── stdlib ───────────────────────────────────────────────────────────
import os, sys, json, time, random, re, hashlib, logging, joblib, argparse
from pathlib import Path
from getpass import getuser
from collections import defaultdict

# ─── third-party ──────────────────────────────────────────────────────
import numpy as np, torch, psycopg2
from sentence_transformers import SentenceTransformer
from torch_geometric.data.data import DataEdgeAttr
import torch.serialization as serialization

# ─── local modules ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.parse_plan  import parse_plan, canonical as _can
from modules.plan2vec    import Plan2VecEncoder
from modules.train_gnimc import GNIMCModel
from modules.eim         import EmbeddingMemory
from modules.bandit      import ThompsonBandit

# ─── CLI flags ────────────────────────────────────────────────────────
argp = argparse.ArgumentParser()
argp.add_argument("--sql-dir", required=True, help="Directory containing *.sql files")
argp.add_argument("--out",     default="./results", help="Where to write run log")
ARGS = argp.parse_args()



RAW_SQL_DIR = Path(ARGS.sql_dir).expanduser()
OUT_DIR     = Path(ARGS.out).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_F = OUT_DIR / "baseline.json"
try:
    baseline_map = json.load(open(BASELINE_F))
except FileNotFoundError:
    baseline_map = {}

# EMA calibration (if you prefer statistical correction)
calib = {"alpha": 0.1, "c": 1.0}


# ─── reproducibility & logging ────────────────────────────────────────
torch.manual_seed(0); np.random.seed(0); random.seed(0)
serialization.add_safe_globals([DataEdgeAttr])

log_lvl = logging.DEBUG if os.getenv("IMC_DEBUG") else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(levelname)s: %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(OUT_DIR / "run.log", mode="w")
                    ])
L = logging.getLogger("IMC")

# ─── constant paths / artefacts ───────────────────────────────────────
MODELS_DIR = Path.home() / "models"
DATA_ROOT  = Path.home() / "New_Data"
PLAN_ROOT  = Path.home() / "Downloads" / "dsb"

HINT_IDS  = json.load(open(MODELS_DIR / "hint_ids.json"))
SQL2IDX   = json.load(open(MODELS_DIR / "sql2idx.json"))
PIPELINE  = joblib.load(MODELS_DIR / "pipeline.pkl")
OP_MAP    = json.load(open(MODELS_DIR / "op_map.json"))

SQL_VOCAB = json.load(open(MODELS_DIR / "sql_vocab.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── helper mats & constant tensors ───────────────────────────────────
embed_train = torch.from_numpy(
    PIPELINE.transform(
        np.load("/Users/raahimlone/rahhh/new/embeddings/syntaxA_embedding.npy")
    )
).float().to(DEVICE)

Y_scaled = np.load(DATA_ROOT / "Y_scalednew.npz")["Y"]
H_ALL    = torch.from_numpy(Y_scaled).float().to(DEVICE)

W, M = np.load(DATA_ROOT / "Wnew.npy"), np.load(DATA_ROOT / "Mnew.npy")
MU_Y, SIG_Y = np.log1p(W[M > 0]).mean(), np.log1p(W[M > 0]).std()

SQL_TOK_MAXLEN = 128
HINT_TARGETS   = ["hashjoin","indexonlyscan","indexscan",
                  "mergejoin","nestloop","seqscan"]

# ─── build models (with auto edge-embed resize) ───────────────────────
def _build_models():
    sample_plan = parse_plan(
        json.load(open(next(PLAN_ROOT.rglob("*.json"))))
    )
    numeric_dim = sample_plan.x.size(1) - 1

    enc = Plan2VecEncoder(
        num_op_types=len(OP_MAP),
        numeric_dim=numeric_dim,
        vocab_size=len(SQL_VOCAB),
        text_dim=64,
        hidden_dim=256,
        num_layers=3,
        out_dim=512,
    ).to(DEVICE)

    head = GNIMCModel(
        q_dim=enc.out_dim + 120,
        h_dim=H_ALL.size(1),
        rank=128,
        num_hints=len(HINT_IDS),
    ).to(DEVICE)

    enc_ck  = torch.load(MODELS_DIR / "plan2vec_ckpt.pt", map_location=DEVICE)
    head_ck = torch.load(MODELS_DIR / "gnimc_ckpt.pt",    map_location=DEVICE)

    # --- edge_embed size mismatch patch
    if enc_ck["edge_embed.weight"].size(0) != enc.edge_embed.weight.size(0):
        new_sz   = enc_ck["edge_embed.weight"].size(0)
        emb_dim  = enc.edge_embed.weight.size(1)
        enc.edge_embed = torch.nn.Embedding(new_sz, emb_dim).to(DEVICE)

    enc.load_state_dict(enc_ck, strict=False)
    head.load_state_dict(head_ck["model_state_dict"])
    enc.eval(); head.eval()
    return enc, head

ENC, HEAD = _build_models()

# ─── memory + bandit ──────────────────────────────────────────────────
EIM    = EmbeddingMemory(dim=ENC.out_dim + 120, tau=8.0)
BANDIT = ThompsonBandit()

# ─── misc helpers ─────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def _pca_vec(sql: str) -> torch.Tensor:
    can = _can(sql)
    if can in SQL2IDX:
        return embed_train[SQL2IDX[can]: SQL2IDX[can] + 1]
    vec = embedder.encode([sql], normalize_embeddings=True)
    return torch.from_numpy(PIPELINE.transform(vec)).float().to(DEVICE)

def _sql_tokens(sql: str):
    toks = re.findall(r"\w+", sql.lower())[:SQL_TOK_MAXLEN]
    ids  = [SQL_VOCAB.get(t, 1) for t in toks] + [0]*(SQL_TOK_MAXLEN-len(toks))
    mask = [1]*len(toks) + [0]*(SQL_TOK_MAXLEN-len(toks))
    return (
        torch.tensor([ids],  dtype=torch.long,   device=DEVICE),
        torch.tensor([mask], dtype=torch.float32,device=DEVICE),
    )

def _hash(sql: str) -> str:
    return hashlib.sha1(sql.encode()).hexdigest()

def _toggles_from_hint(hint_combo: str) -> str:
    if not hint_combo:          # baseline – no changes
        return ""
    parts = hint_combo.split(",")
    return " ".join(
        f"SET enable_{h}={'on' if h in parts else 'off'};"
        for h in HINT_TARGETS
    )

# ─── Postgres connection ──────────────────────────────────────────────
conn = psycopg2.connect(
    dbname="IMDB",
    user=getuser(),
    host="localhost",
    port="6543",
    password=os.getenv("PGPASSWORD", ""),
)
conn.autocommit = True
cur = conn.cursor()

def _parse_explain(raw):
    obj = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
    return obj[0] if isinstance(obj, list) else obj

# ─── inference routine ────────────────────────────────────────────────
def choose_hint(sql_text: str):
    """Return hint_combo, z_vec, qkey, chosen_arm."""
    qkey = _hash(sql_text)

    # –– embed z
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql_text}")
    g = parse_plan(_parse_explain(cur.fetchone()[0])).to(DEVICE)
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=DEVICE)
    sql_ids, sql_mask = _sql_tokens(sql_text)

    with torch.no_grad():
        z_core = ENC(g, sql_ids, sql_mask)
    z = torch.cat([z_core, _pca_vec(sql_text)], dim=1)

    # –– candidate arms
    arms, mu_sigma = {}, {}
    # index of the empty‐string hint == “no hints” (default plan)
    with torch.no_grad():
        mu, sig, _ = HEAD(z, H_ALL)
    mu_np, sig_np = mu.cpu().numpy(), sig.cpu().numpy()
    lat_mu = np.expm1((mu_np * SIG_Y) + MU_Y) * 1e3  # predicted [ms]

    # derive default plan index & its predicted latency
    default_idx  = HINT_IDS.index("")       # empty string = no hints
    base_pred_ms = float(lat_mu[0])         # μ[0] = default‐plan prediction

    # 2-A cache neighbour (only if history < model’s default prediction)
    n_idx, n_lat = EIM.query(z.cpu().numpy())
    if n_idx is not None and n_idx != default_idx and n_lat < base_pred_ms:
        arms["cache"]     = (n_idx, n_lat)
        mu_sigma["cache"] = (float(n_lat), 0.0)

    # 2-B best-of-knn (ditto)
    kn_idx, kn_lat = EIM.best_of_knn(z.cpu().numpy(), k=10)
    # require at least 8 stored points *and* ≥5% predicted improvement
    if (
        kn_idx is None
        or len(EIM._vals) < 8
        or kn_lat >= base_pred_ms * 0.95
    ):
        kn_idx, kn_lat = None, None
    if kn_idx is not None and kn_idx != default_idx and kn_lat < base_pred_ms:
        arms["knn"]       = (kn_idx, kn_lat)
        mu_sigma["knn"]   = (float(kn_lat), 0.0)

    base_pred_ms = float(lat_mu[0])          # μ for the default plan
    default_idx  = HINT_IDS.index("")        # empty string = no hints
    arms["default"]     = (default_idx, base_pred_ms)        #  ← NEW
    mu_sigma["default"] = (base_pred_ms, 0.0)                #  ← NEW
    K = 5
    for rank, idx in enumerate(np.argsort(lat_mu)[:K], 1):
        arm_name = f"imc{rank}"
        penalty  = 1 + 0.5*sig_np[idx]
        arms[arm_name]     = (int(idx), float(lat_mu[idx]*penalty))
        mu_sigma[arm_name] = (float(lat_mu[idx]), float(sig_np[idx]))

    base_pred_ms = float(lat_mu[0])       # idx 0 == default

    if qkey in BEST_HINT:
        arms["best"] = BEST_HINT[qkey]

    arm = BANDIT.choose(
        qkey,
        arms,
        base_pred_ms,
        mu_sigma,
        actual_base_ms=true_base_ms,
        z=z.cpu().numpy(),
    )
    hint_idx = arms[arm][0]
    return HINT_IDS[hint_idx], z, qkey, arm, base_pred_ms

# ─── global state ─────────────────────────────────────────────────────
BEST_HINT          : dict[str, tuple[int,float]] = {}

# ─── driver loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        for sql_file in sorted(RAW_SQL_DIR.rglob("*.sql")):
            sql_text = sql_file.read_text()
            qkey     = _hash(sql_text)

            cnt = baseline_map.get(qkey, {}).get("count", 0) + 1
            if cnt == 1 or cnt % 100 == 0:
                t0 = time.time()
                cur.execute(sql_text)
                b_ms = (time.time() - t0)*1e3
                baseline_map[qkey] = {"lat": b_ms, "count": cnt}
                with open(BASELINE_F, "w") as f:
                    json.dump(baseline_map, f)
            true_base_ms = baseline_map[qkey]["lat"] * calib["c"]


            # choose hint
            hint_combo, z_vec, qkey, arm, base_pred_ms = choose_hint(sql_text)

            # execute with hint
            toggles = _toggles_from_hint(hint_combo)
            t0 = time.time()
            cur.execute(toggles + " " + sql_text)
            lat_ms = (time.time() - t0)*1e3

            # feedback
            idx_hint = HINT_IDS.index(hint_combo)
            ok = lat_ms <= true_base_ms * 1.05

            EIM.update_stats(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
            if ok:                               # only store good outcomes
                EIM.add(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
            BANDIT.update(
                qkey,
                arm,
                lat_ms,
                true_base_ms,
                z=z_vec.cpu().numpy(),
            )

            if ok and (qkey not in BEST_HINT or lat_ms < BEST_HINT[qkey][1]):
                BEST_HINT[qkey] = (idx_hint, lat_ms)

            msg = (f"{sql_file.name:20s}  arm={arm:5s} "
                   f"hints=({hint_combo or 'none':30s})  "
                   f"lat={lat_ms:7.2f} ms   base={base_pred_ms:7.2f}")
            L.info(msg)

    finally:
        cur.close(); conn.close()
