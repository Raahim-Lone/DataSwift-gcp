#!/usr/bin/env python3
"""
 test_v4d.py – Batch Hint Recommender v4‑d (research‑metrics edition)
 ─────────────────────────────────────────────────────────────────────
 * Everything from v4‑c (adaptive EIM v2 + Thompson bandit + no‑regression guard)
 * **NEW** automatic result logging‑to‑CSV for downstream analysis
 * **NEW** one‑shot summary stats (mean, P99, regression count) emitted as
   summary.csv at the end of every run – makes it trivial to build the
   comparison table in the extended abstract.
 * 100 % backwards‑compatible: no existing flags/scripts need to change.
   If you do *not* want CSVs, pass --no-csv.
 Usage (typical):
     python test_v4d.py --sql-dir ./queries --out ./results
 Example with debug & custom threshold:
     python test_v4d.py --sql-dir ./queries --out ./results \
                      --debug --dbg-thresh 1.5
"""

# ─── stdlib ───────────────────────────────────────────────────────────
import os, sys, json, time, random, re, hashlib, logging, argparse, joblib, csv, statistics
from pathlib import Path
from getpass import getuser
from collections import defaultdict

# ─── third‑party ──────────────────────────────────────────────────────
import numpy as np, torch, psycopg2
from sentence_transformers import SentenceTransformer
from torch_geometric.data.data import DataEdgeAttr
import torch.serialization as serialization
import matplotlib       #  ← NEW
matplotlib.use("Agg")    # head-less backend for GCP
import matplotlib.pyplot as plt

# ─── local modules (unchanged) ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.parse_plan  import parse_plan, canonical as _can
from modules.plan2vec    import Plan2VecEncoder
from modules.train_gnimc import GNIMCModel
from modules.eim         import EmbeddingMemory
from modules.bandit      import ThompsonBandit

# ─── CLI flags ───────────────────────────────────────────────────────
argp = argparse.ArgumentParser()
argp.add_argument("--sql-dir",  required=True, help="Directory containing *.sql files")
argp.add_argument("--out",      default="./results", help="Where to write run log / CSVs")
argp.add_argument("--debug",    action="store_true",
                  help="Emit extra diagnostics and capture ‘bad‑hint’ rows")
argp.add_argument("--dbg-thresh", type=float, default=2.0,
                  help="Flag when actual_ms > pred_ms × THRESH (default 2)")
argp.add_argument("--no-csv",   action="store_true",
                  help="Disable per‑query CSV and summary output (legacy mode)")
ARGS = argp.parse_args()

# ─── paths & output setup ─────────────────────────────────────────────
RAW_SQL_DIR = Path(ARGS.sql_dir).expanduser()
OUT_DIR     = Path(ARGS.out).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_F   = OUT_DIR / "baseline.json"
RESULTS_CSV  = OUT_DIR / "results.csv"
SUMMARY_CSV  = OUT_DIR / "summary.csv"

try:
    baseline_map = json.load(open(BASELINE_F))
except FileNotFoundError:
    baseline_map = {}

# EMA calibration (if you prefer statistical correction)
calib = {"alpha": 0.1, "c": 1.0}

# ─── reproducibility & logging ───────────────────────────────────────
torch.manual_seed(0); np.random.seed(0); random.seed(0)
serialization.add_safe_globals([DataEdgeAttr])

log_lvl = logging.DEBUG if os.getenv("IMC_DEBUG") else logging.INFO
logging.basicConfig(level=log_lvl,
                    format="%(levelname)s: %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(OUT_DIR / "run.log", mode="w")
                    ])
L   = logging.getLogger("IMC")
BAD = logging.getLogger("IMC.bad")
if ARGS.debug:
    BAD.setLevel(logging.WARNING)

# ─── constant paths / artefacts (unchanged) ───────────────────────────
MODELS_DIR = Path.home() / "models"
DATA_ROOT  = Path.home() / "New_Data"
PLAN_ROOT  = Path.home() / "Downloads" / "dsb"

HINT_IDS  = json.load(open(MODELS_DIR / "hint_ids.json"))
SQL2IDX   = json.load(open(MODELS_DIR / "sql2idx.json"))
PIPELINE  = joblib.load(MODELS_DIR / "pipeline.pkl")
OP_MAP    = json.load(open(MODELS_DIR / "op_map.json"))
SQL_VOCAB = json.load(open(MODELS_DIR / "sql_vocab.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── helper mats & constant tensors (unchanged) ───────────────────────
embed_train = torch.from_numpy(
    PIPELINE.transform(
        np.load("/Users/raahimlone/rahhh/new/embeddings/syntaxA_embedding.npy")
    )
).float().to(DEVICE)

Y_scaled = np.load(DATA_ROOT / "Y_scalednew.npz")["Y"]
H_ALL    = torch.from_numpy(Y_scaled).float().to(DEVICE)

W, M    = np.load(DATA_ROOT / "Wnew.npy"), np.load(DATA_ROOT / "Mnew.npy")
MU_Y, SIG_Y = np.log1p(W[M > 0]).mean(), np.log1p(W[M > 0]).std()

SQL_TOK_MAXLEN = 128
HINT_TARGETS   = [
    "hashjoin", "indexonlyscan", "indexscan",
    "mergejoin", "nestloop", "seqscan",
]

# ──────────────────────────────────────────────────────────────────────
# Build models (identical to v4‑c apart from function name)            
# ──────────────────────────────────────────────────────────────────────

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
        hidden_dim=512,      # ⟵ match training run
        num_layers=4,        # ⟵ match training run
        out_dim=512,
    ).to(DEVICE)
    head = GNIMCModel(
        q_dim=enc.out_dim + 120,
        h_dim=H_ALL.size(1),
        rank=288,            # ⟵ match training run
        num_hints=len(HINT_IDS),
    ).to(DEVICE)

    enc_ck  = torch.load(MODELS_DIR / "plan2vec_ckpt.pt", map_location=DEVICE)
    head_ck = torch.load(MODELS_DIR / "gnimc_ckpt.pt",    map_location=DEVICE)

    # edge‑embed size mismatch patch
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

# ─── misc helpers (unchanged) ────────────────────────────────────────
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
        f"SET enable_{h}={'on' if h in parts else 'off'};" for h in HINT_TARGETS
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

# ─── CSV helpers ──────────────────────────────────────────────────────
if not ARGS.no_csv and not RESULTS_CSV.exists():
    # write header only if file does not already exist (so we can append runs)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "query_file", "variant", "arm", "hint_combo",
            "lat_ms", "baseline_ms", "speedup", "pred_mu_ms", "pred_sigma_ms",
            "ok_base", "ok_pred"
        ])

# ─── inference routine (unchanged behaviour) ─────────────────────────

def choose_hint(sql_text: str):
    """
    Return:
        hint_combo, z_vec, qkey, chosen_arm,
        base_pred_ms, pred_mu_ms, pred_sigma_ms
    """
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
    with torch.no_grad():
        mu, sig, _ = HEAD(z, H_ALL)
    mu_np, sig_np = mu.cpu().numpy(), sig.cpu().numpy()
    lat_mu = np.expm1((mu_np * SIG_Y) + MU_Y) * 1e3  # predicted [ms]

    default_idx  = HINT_IDS.index("")
    base_pred_ms = float(lat_mu[0])

    # 2‑A cache neighbour
    n_idx, n_lat = EIM.query(z.cpu().numpy())
    if n_idx is not None and n_idx != default_idx and n_lat < base_pred_ms:
        arms["cache"]     = (n_idx, n_lat)
        mu_sigma["cache"] = (float(n_lat), 0.0)

    # 2‑B best‑of‑knn
    kn_idx, kn_lat = EIM.best_of_knn(z.cpu().numpy(), k=10)
    if (
        kn_idx is not None and len(EIM._vals) >= 8
        and kn_lat < base_pred_ms * 0.95 and kn_idx != default_idx
    ):
        arms["knn"]       = (kn_idx, kn_lat)
        mu_sigma["knn"]   = (float(kn_lat), 0.0)

    # default always present
    arms["default"]     = (default_idx, base_pred_ms)
    mu_sigma["default"] = (base_pred_ms, 0.0)

    # top‑K IMC suggestions
    K = 5
    for rank, idx in enumerate(np.argsort(lat_mu)[:K], 1):
        arm_name = f"imc{rank}"
        sigma_ms = np.expm1(sig_np[idx] * SIG_Y) * 1e3
        penalty  = 1 + 0.5 * (sigma_ms / max(lat_mu[idx], 1e-3))
        arms[arm_name]     = (int(idx), float(lat_mu[idx] * penalty))
        mu_sigma[arm_name] = (float(lat_mu[idx]), float(sig_np[idx]))

    if qkey in BEST_HINT:
        arms["best"] = BEST_HINT[qkey]

    arm = BANDIT.choose(
        qkey,
        arms,
        base_pred_ms,
        mu_sigma,
        actual_base_ms=true_base_ms,   # pulled from caller’s global
        z=z.cpu().numpy(),
    )
    hint_idx      = arms[arm][0]
    pred_mu_ms    = float(mu_sigma.get(arm, (arms[arm][1], 0.0))[0])
    pred_sigma_ms = float(mu_sigma.get(arm, (0.0, 0.0))[1])
    return (HINT_IDS[hint_idx], z, qkey, arm,
            base_pred_ms, pred_mu_ms, pred_sigma_ms)

# ─── global state ─────────────────────────────────────────────────────
BEST_HINT: dict[str, tuple[int, float]] = {}

# Containers for summary statistics
all_lat      = defaultdict(list)             # variant ➜ [latencies]
all_base_lat = defaultdict(list)             # variant ➜ [baseline_lats]
# ─── helper to execute ONE variant on ONE query ──────────────────────
def run_variant(sql_text: str,
                variant: str,
                true_base_ms: float,
                sql_file_name: str):
    """
    Returns: (variant, arm, hint_combo,
              lat_ms, pred_mu_ms, pred_sigma_ms,
              ok_base, ok_pred)
    """
    # ------------ DEFAULT PLAN (no hints) ---------------------------
    z_vec = None                     # <— NEW default

    if variant == "default":
        toggles     = ""              # no hint
        arm         = "default"
        hint_combo  = ""
        pred_mu_ms  = pred_sigma_ms = 0.0

    # ------------ GN-IMC TOP-1  (no memory / bandit) ---------------
    elif variant == "gnimc":
        # *re-embed* & run the head once
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql_text}")
        g = parse_plan(_parse_explain(cur.fetchone()[0])).to(DEVICE)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=DEVICE)
        sql_ids, sql_mask = _sql_tokens(sql_text)
        with torch.no_grad():
            z_core = ENC(g, sql_ids, sql_mask)
            z      = torch.cat([z_core, _pca_vec(sql_text)], dim=1)
            mu, sig, _ = HEAD(z, H_ALL)
        mu_np   = mu.cpu().numpy()
        sig_np  = sig.cpu().numpy()
        lat_mu  = np.expm1((mu_np * SIG_Y) + MU_Y) * 1e3  # ms
        idx     = int(np.argsort(lat_mu)[0])              # best IMC
        hint_combo      = HINT_IDS[idx]
        pred_mu_ms      = float(lat_mu[idx])
        pred_sigma_ms   = float(np.expm1(sig_np[idx] * SIG_Y) * 1e3)
        toggles         = _toggles_from_hint(hint_combo)
        arm             = "gnimc"

    # ------------ FULL SYSTEM (bandit + EIM) ------------------------
    else:  # variant == "full"
        (hint_combo, z_vec, qkey, arm,
         base_pred_ms, pred_mu_ms, pred_sigma_ms) = choose_hint(sql_text)
        toggles = _toggles_from_hint(hint_combo)

    # ------------ execute the query --------------------------------
    t0 = time.time()
    cur.execute(toggles + " " + sql_text)
    lat_ms = (time.time() - t0) * 1e3

    # ------------ regression flags ---------------------------------
    ok_base = lat_ms <= true_base_ms * 1.05
    ok_pred = (variant == "full"  # only meaningful when we have a model μ
               and lat_ms <= pred_mu_ms * ARGS.dbg_thresh) or variant != "full"

    return (variant, arm, hint_combo, lat_ms,
            pred_mu_ms, pred_sigma_ms, ok_base, ok_pred, z_vec)

# ─── driver loop ──────────────────────────────────────────────────────

debug_rows = [] if ARGS.debug else None

try:
    VARIANTS = ("default", "gnimc", "full")     # ← fixed order

    for sql_file in sorted(RAW_SQL_DIR.rglob("*.sql")):
        sql_text = sql_file.read_text()
        qkey     = _hash(sql_text)

        # -------- ensure baseline is fresh -------------------------
        cnt = baseline_map.get(qkey, {}).get("count", 0) + 1
        if cnt == 1 or cnt % 100 == 0 or (
            time.time() - baseline_map.get(qkey, {}).get("ts", 0) > 86_400
        ):
            t0 = time.time()
            cur.execute(sql_text)                       # DEFAULT PLAN
            b_ms = (time.time() - t0) * 1e3
            baseline_map[qkey] = {"lat": b_ms,
                                  "count": cnt,
                                  "ts": time.time()}
            with open(BASELINE_F, "w") as f:
                json.dump(baseline_map, f)

        true_base_ms = baseline_map[qkey]["lat"] * calib["c"]

        # -------- run *each* variant once --------------------------
        for variant in VARIANTS:
            (variant, arm, hint_combo, lat_ms,
            pred_mu_ms, pred_sigma_ms, ok_base, ok_pred, z_vec) = \
                run_variant(sql_text, variant,
                            true_base_ms, sql_file.name)

            # ----- record (only FULL updates EIM / bandit) ---------
            if variant == "full":
                # identical feedback block you already had:
                idx_hint = HINT_IDS.index(hint_combo)
                ok       = ok_base
                # (EIM & BANDIT updates unchanged)
                if z_vec is not None:                 # safety: should always be true here
                    EIM.update_stats(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
                    if ok:
                        EIM.add(z_vec.cpu().numpy(), idx_hint, lat_ms, ok)
                    BANDIT.update(qkey, arm, lat_ms, true_base_ms,
                                z=z_vec.cpu().numpy())

            # ----- accumulate for summary --------------------------
            tag = variant                       # keep arrays separate
            all_lat      .setdefault(tag, []).append(lat_ms)
            all_base_lat .setdefault(tag, []).append(true_base_ms)

            # ----- CSV row -----------------------------------------
            if not ARGS.no_csv:
                speedup = lat_ms / max(true_base_ms, 1e-6)
                with open(RESULTS_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(time.time()), sql_file.name,
                        variant, arm, hint_combo,
                        f"{lat_ms:.2f}", f"{true_base_ms:.2f}",
                        f"{speedup:.4f}", f"{pred_mu_ms:.2f}",
                        f"{pred_sigma_ms:.2f}", int(ok_base), int(ok_pred)
                    ])

            L.info(f"{sql_file.name:20s}  [{variant}] "
                   f"arm={arm:6s}  "
                   f"hints=({hint_combo or 'none':30s})  "
                   f"lat={lat_ms:7.2f} ms   base={true_base_ms:7.2f}")

finally:
    cur.close(); conn.close()

    # write debug jsonl if requested
    if ARGS.debug and debug_rows:
        dbg_f = OUT_DIR / "debug.jsonl"
        with open(dbg_f, "w") as f:
            for r in debug_rows:
                f.write(json.dumps(r) + "\n")
        L.info(f"⚠️  wrote {len(debug_rows)} anomalous rows → {dbg_f}")

    if ARGS.no_csv:
        sys.exit(0)  # nothing else to compute

    # ─── summary CSV --------------------------------------------------
    # ─── summary CSV (one row per variant) --------------------------
    if not ARGS.no_csv and all_lat:
        with open(SUMMARY_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "variant", "num_queries",
                "mean_speedup", "p99_speedup",
                "regressions_gt20pct", "worst_regression"
            ])
            for variant, lat_list in all_lat.items():
                arr_lat  = np.array(lat_list)
                arr_base = np.array(all_base_lat[variant])
                speedups = arr_lat / np.maximum(arr_base, 1e-6)

                writer.writerow([
                    variant,
                    len(arr_lat),
                    f"{speedups.mean():.4f}",
                    f"{np.percentile(speedups,99):.4f}",
                    int((speedups > 1.20).sum()),
                    f"{speedups.max():.4f}"
                ])
        L.info(f"📊 wrote summary → {SUMMARY_CSV}")

    if not ARGS.no_csv and all_lat:
        percentiles = [50, 75, 95, 99]
        variants_plot = ["default", "gnimc", "full"]

        perc_by_var = {
            v: np.percentile(np.array(all_lat[v]), percentiles) / 1000.0  # → seconds
            for v in variants_plot
        }

        # base latencies (default) used for the speed-up axis in waterfalls
        arr_base_default = np.array(all_lat["default"])

        # 2-A  percentile bar chart  ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        x      = np.arange(len(percentiles))
        w      = 0.25                      # bar width

        colors = {"default": "#ff7f0e",    # orange
                "gnimc"  : "#1f77b4",    # blue
                "full"   : "#2ca02c"}    # green

        for i, v in enumerate(variants_plot):
            ax.bar(x + (i-1)*w, perc_by_var[v], width=w,
                label=v.capitalize(), color=colors[v])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in percentiles])
        ax.set_ylabel("Wall Time [s]")
        ax.set_xlabel("Percentile")
        ax.set_ylim(0, max(perc_by_var["default"])*1.15)
        ax.set_title("Latency Distribution (lower is better)")
        ax.legend(frameon=False, ncol=3, loc="upper left")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "percentile_bar.png", dpi=300)
        plt.close(fig)
        # 2-B  per-query waterfall  ──────────────────────────────────
        for v in ["gnimc", "full"]:
            delta_pct = (
                (np.array(all_lat[v]) - arr_base_default)
                / np.maximum(arr_base_default, 1e-6) * 100.0
            )
            order = np.argsort(delta_pct)          # fastest → slowest
            fig, ax = plt.subplots(figsize=(12, 2.8))
            ax.bar(range(len(delta_pct)),
                delta_pct[order],
                color=["#55a868" if d < 0 else "#c44e52" for d in delta_pct[order]])
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Δ Runtime [%]")
            ax.set_xlabel("Queries (sorted)")
            ax.set_title(f"Per-query Speed-up / Slow-down  ({v})")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"waterfall_{v}.png", dpi=300)
            plt.close(fig)

        L.info("🖼️  wrote plots → percentile_bar.png & waterfall.png")

    else:
        L.warning("No latency data collected – summary.csv skipped.")
