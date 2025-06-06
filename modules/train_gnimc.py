#!/usr/bin/env python3
"""
train_gnimc_matrix.py — two-phase UV training, ranking + classification
────────────────────────────────────────────────────────────────────────
Fixes (2025-05-21)
• Writes models/sql2idx.json   (canonical SQL → row-index)
• Writes models/pipeline.pkl   (scaler+PCA) for test-time reuse
• Corrects one hint-ID typo (“seqcan”→“seqscan”), de-dups and sorts list
"""
from __future__ import annotations

# ─── stdlib ────────────────────────────────────────────────────────────
import os, sys, json, time, random, re, joblib
from pathlib import Path

# ─── third-party ───────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MarginRankingLoss, CrossEntropyLoss

# ─── local imports ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.parse_plan import parse_plan, canonical, _unwrap_plan
from modules.op_maps    import NODE_TYPE_MAP, EDGE_TYPE_MAP
from modules.plan2vec   import Plan2VecEncoder

# ───────────────────────────── metrics ────────────────────────────────
def ranking_accuracy(mu: torch.Tensor, y: torch.Tensor) -> float:
    """Pairwise accuracy for ranking metric."""
    B = mu.size(0)
    if B < 2:
        return float("nan")
    idx   = torch.randperm(B, device=mu.device)
    half  = B // 2
    i1, i2 = idx[:half], idx[half : 2 * half]
    y1, y2 = y[i1], y[i2]
    m1, m2 = mu[i1], mu[i2]
    corr = ((y1 < y2) & (m1 < m2)) | ((y1 > y2) & (m1 > m2))
    return corr.float().mean().item()


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ───────────────────────── helper builders ────────────────────────────
def warm_up_maps(plan_root: Path) -> None:
    """Populate NODE_TYPE_MAP & EDGE_TYPE_MAP from all plan JSONs."""
    for path in plan_root.rglob("*.json"):
        recs = json.load(open(path))
        recs = recs if isinstance(recs, list) else [recs]
        for rec in recs:
            plan  = _unwrap_plan(rec)
            stack = [plan]
            while stack:
                node = stack.pop()
                op   = canonical(node.get("Node Type", "Unknown"))
                NODE_TYPE_MAP.setdefault(op, len(NODE_TYPE_MAP))
                jt   = node.get("Join Type", "child")
                EDGE_TYPE_MAP.setdefault(jt, len(EDGE_TYPE_MAP))
                for c in node.get("Plans", []):
                    stack.append(c)

    # ensure unknown=0
    if "unknown" not in NODE_TYPE_MAP:
        NODE_TYPE_MAP["unknown"] = 0
    if NODE_TYPE_MAP["unknown"] != 0:
        ordered = ["unknown"] + [k for k in NODE_TYPE_MAP if k != "unknown"]
        NODE_TYPE_MAP.clear()
        for i, op in enumerate(ordered):
            NODE_TYPE_MAP[op] = i


def rebuild_sqls(plan_root: Path):
    sqls, mapping = [], {}
    for path in plan_root.rglob("*.json"):
        recs = json.load(open(path))
        recs = recs if isinstance(recs, list) else [recs]
        for rec in recs:
            q = rec.get("sql", "").strip()
            if q and q not in mapping:
                mapping[q] = str(path)
                sqls.append(q)
    return sqls, mapping


def build_sql_vocab(plan_root: Path, max_size: int = 10_000):
    tok2idx = {"<pad>": 0, "<unk>": 1}
    tok = re.compile(r"\w+").findall
    for path in plan_root.rglob("*.json"):
        recs = json.load(open(path))
        recs = recs if isinstance(recs, list) else [recs]
        for rec in recs:
            for t in tok(rec.get("sql", "")):
                if t not in tok2idx:
                    tok2idx[t] = len(tok2idx)
                    if len(tok2idx) >= max_size:
                        return tok2idx
    return tok2idx


# ───────────────────────────── dataset ────────────────────────────────
class PairDS(Dataset):
    def __init__(
        self,
        obs,
        sqls,
        sql2path,
        W,
        Y,
        mu,
        sig,
        sql_vocab: dict[str, int],
        maxlen: int = 128,
    ):
        self.obs, self.sqls, self.sql2path = obs, sqls, sql2path
        self.W, self.Y, self.mu, self.sig = W, Y, mu, sig
        self.vocab, self.maxlen = sql_vocab, maxlen

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, k):
        q_idx, h_idx = self.obs[k]
        recs = json.load(open(self.sql2path[self.sqls[q_idx]]))
        rec  = recs[0] if isinstance(recs, list) else recs
        g    = parse_plan(rec)
        h    = torch.from_numpy(self.Y[h_idx]).float()
        y    = (np.log1p(self.W[q_idx, h_idx]) - self.mu) / self.sig

        toks = re.findall(r"\w+", rec.get("sql", "").lower())
        ids  = [self.vocab.get(t, 1) for t in toks[: self.maxlen]]
        mask = [1] * len(ids)
        pad  = self.maxlen - len(ids)
        if pad > 0:
            ids += [0] * pad
            mask += [0] * pad

        return (
            q_idx,
            g,
            torch.tensor(ids),
            torch.tensor(mask),
            h,
            torch.tensor(y),
        )


def collate(batch):
    qs, gs, sql_ids, sql_mask, hs, ys = zip(*batch)
    return (
        torch.tensor(qs, dtype=torch.long),
        Batch.from_data_list(gs),
        torch.stack(sql_ids),
        torch.stack(sql_mask),
        torch.stack(hs),
        torch.stack(ys),
    )


# ───────────────────────────── model(s) ───────────────────────────────
class GNIMCModel(nn.Module):
    def __init__(self, q_dim: int, h_dim: int, rank: int, num_hints: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(q_dim, rank) * 0.2)
        self.V = nn.Parameter(torch.randn(h_dim, rank) * 0.2)

        self.bias_z  = nn.Linear(1, 1, bias=True)
        self.bias_h  = nn.Linear(1, 1, bias=True)
        self.var_raw = nn.Linear(2 * rank + 1, 1)
        nn.init.constant_(self.var_raw.bias, float(np.log(np.expm1(1.0))))
        nn.init.constant_(self.var_raw.weight, 0.0)

        self.classifier = nn.Linear(q_dim, num_hints)

    def forward(self, z, h):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.size(0) != h.size(0):
            if z.size(0) == 1:
                z = z.expand(h.size(0), -1)
            else:
                h = h.expand(z.size(0), -1)

        u, v = z @ self.U, h @ self.V
        inter = (u * v).sum(1)
        ones  = torch.ones_like(inter.unsqueeze(-1))
        mu    = inter + self.bias_z(ones).squeeze(-1) + self.bias_h(ones).squeeze(-1)

        phi = torch.cat([u, v, (u - v).abs().sum(1, keepdim=True)], dim=1)
        sig = nn.functional.softplus(self.var_raw(phi)).squeeze(-1) + 1e-4

        logits = self.classifier(z)
        return mu, sig, logits

    @staticmethod
    def nll(mu, sig, y):
        mu = torch.clamp(mu, -8, 8)
        return (torch.log(sig) + (y - mu).pow(2) / (2 * sig.pow(2))).mean()


# ──────────────────────────── training ────────────────────────────────
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    t0 = time.time()

    # 1) runtime matrices -------------------------------------------------
    data_root = Path.home() / "New_Data"
    W = np.load(data_root / "Wnew.npy")
    M = np.load(data_root / "Mnew.npy")
    Y = np.load(data_root / "Y_scalednew.npz")["Y"]
    Y = StandardScaler().fit_transform(Y)
    num_q, num_h = W.shape

    # 2) plan corpus, vocab, maps ----------------------------------------
    plan_root = Path.home() / "Downloads" / "dsb"
    warm_up_maps(plan_root)
    print("✅ operator map size:", len(NODE_TYPE_MAP))
    sql_vocab = build_sql_vocab(plan_root)

    models_dir = Path.home() / "models"
    models_dir.mkdir(exist_ok=True)
    json.dump(NODE_TYPE_MAP, open(models_dir / "op_map.json", "w"), indent=2)
    json.dump(EDGE_TYPE_MAP, open(models_dir / "edge_map.json", "w"), indent=2)
    json.dump(sql_vocab,     open(models_dir / "sql_vocab.json", "w"), indent=2)

    # 3) hint-ID list (fix typo, dedup, sort) ----------------------------
    hint_ids_raw = [
        "", "hashjoin,indexonlyscan","hashjoin,indexonlyscan,indexscan",
        "hashjoin,indexonlyscan,indexscan,mergejoin",
        "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop",
        "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
        "hashjoin,indexonlyscan,indexscan,nestloop",
        "hashjoin,indexonlyscan,indexscan,nestloop,seqscan",
        "hashjoin,indexonlyscan,indexscan,seqscan",
        "hashjoin,indexonlyscan,mergejoin",
        "hashjoin,indexonlyscan,mergejoin,nestloop",
        "hashjoin,indexonlyscan,mergejoin,nestloop,seqscan",
        "hashjoin,indexonlyscan,mergejoin,seqscan",
        "hashjoin,indexonlyscan,nestloop",
        "hashjoin,indexonlyscan,nestloop,seqscan",
        "hashjoin,indexonlyscan,seqscan",
        "hashjoin,indexscan","hashjoin,indexscan,mergejoin",
        "hashjoin,indexscan,mergejoin,nestloop",
        "hashjoin,indexscan,mergejoin,nestloop,seqscan",
        "hashjoin,indexscan,mergejoin,seqcan",      # typo
        "hashjoin,indexscan,nestloop",
        "hashjoin,indexscan,nestloop,seqscan",
        "hashjoin,indexscan,seqscan",
        "hashjoin,mergejoin,nestloop,seqscan",
        "hashjoin,mergejoin,seqscan","hashjoin,nestloop,seqscan",
        "hashjoin,seqscan","indexonlyscan,indexscan,mergejoin",
        "indexonlyscan,indexscan,mergejoin,nestloop",
        "indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
        "indexonlyscan,indexscan,mergejoin,seqscan",
        "indexonlyscan,indexscan,nestloop",
        "indexonlyscan,indexscan,nestloop,seqcan",  # typo
        "indexonlyscan,mergejoin","indexonlyscan,mergejoin,nestloop",
        "indexonlyscan,mergejoin,nestloop,seqscan",
        "indexonlyscan,mergejoin,seqscan","indexonlyscan,nestloop",
        "indexonlyscan,nestloop,seqscan","indexscan,mergejoin",
        "indexscan,mergejoin,nestloop","indexscan,mergejoin,nestloop,seqscan",
        "indexscan,mergejoin,seqscan","indexscan,nestloop",
        "indexscan,nestloop,seqscan","mergejoin,nestloop,seqscan",
        "mergejoin,seqscan","nestloop,seqscan"
    ]
    clean = []
    for combo in hint_ids_raw:
        fixed = combo.replace("seqcan", "seqscan")
        clean.append(",".join(sorted(filter(None, fixed.split(",")))))
    hint_ids = sorted(set(clean))
    json.dump(hint_ids, open(models_dir / "hint_ids.json", "w"), indent=2)

    # 4) LLM embeddings + PCA -------------------------------------------
    embed_path = Path("/Users/raahimlone/rahhh/new/embeddings/syntaxA_embedding.npy")
    if not embed_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    X_np = np.load(embed_path)

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=120))]
    )
    X_pca = pipeline.fit_transform(X_np)
    joblib.dump(pipeline, models_dir / "pipeline.pkl")
    X_pca_t = torch.from_numpy(X_pca).float()  # (num_q,120)

    # 5) canonical-SQL → row-idx mapping --------------------------------
    sqls, sql2path = rebuild_sqls(plan_root)
    assert (
        len(sqls) == num_q
    ), f"#sqls ({len(sqls)}) != num_q in W ({num_q})"
    sql2idx = {canonical(s): i for i, s in enumerate(sqls)}
    json.dump(sql2idx, open(models_dir / "sql2idx.json", "w"), indent=2)

    # 6) dataset / loader ------------------------------------------------
    obs    = np.argwhere(M > 0)
    mu_y   = np.log1p(W[M > 0]).mean()
    sig_y  = np.log1p(W[M > 0]).std()
    ds     = PairDS(obs, sqls, sql2path, W, Y, mu_y, sig_y, sql_vocab)
    dl     = DataLoader(ds, batch_size=64, shuffle=True,
                        collate_fn=collate, num_workers=4)
    best_labels = torch.from_numpy(np.argmin(W, axis=1)).long()

    # 7) build models ----------------------------------------------------
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numeric_dim = ds[0][1].x.size(1) - 1
    encoder = Plan2VecEncoder(
        num_op_types=len(NODE_TYPE_MAP),
        numeric_dim=numeric_dim,
        vocab_size=len(sql_vocab),
        text_dim=64,
        hidden_dim=256,
        num_layers=3,
        out_dim=512,
    ).to(device)
    X_pca_t = X_pca_t.to(device)

    head = GNIMCModel(
        q_dim=encoder.out_dim + 120,
        h_dim=Y.shape[1],
        rank=128,
        num_hints=len(hint_ids),
    ).to(device)

    #   initialise bias offsets
    head.bias_z.weight.data.zero_()
    head.bias_z.bias.data.fill_(mu_y)
    head.bias_h.weight.data.zero_()
    head.bias_h.bias.data.zero_()

    # 8) losses, optimiser ----------------------------------------------
    ce_fn   = CrossEntropyLoss()
    rank_fn = MarginRankingLoss(margin=0.1)
    opt = AdamW(
        [
            {"params": encoder.parameters(),        "lr": 3e-3},
            {"params": head.U,                      "lr": 1e-3},
            {"params": head.V,                      "lr": 1e-3},
            {"params": head.bias_z.parameters(),    "lr": 1e-1},
            {"params": head.bias_h.parameters(),    "lr": 1e-1},
            {"params": head.var_raw.parameters(),   "lr": 1e-6},
            {"params": head.classifier.parameters(),"lr": 1e-3},
        ],
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    # 9) training loop ---------------------------------------------------
    warmup = 33
    for p in head.var_raw.parameters():
        p.requires_grad = False

    LAMBDA_RANK = 1.0
    LAMBDA_CE   = 0.1

    for ep in range(1, 59):  # 50 epochs
        encoder.train()
        head.train()

        if ep == warmup + 1:
            print(f"-- Epoch {ep}: PHASE 2 (NLL) — unfreezing var_raw --")
            for p in head.var_raw.parameters():
                p.requires_grad = True

        sum_pred = sum_rank = sum_ce = sum_tot = 0.0

        for batch_idx, (q_idx, g, sql_ids, sql_mask, h, y) in enumerate(dl):
            B = y.size(0)

            # move to device
            q_idx    = q_idx.to(device)
            g        = g.to(device)
            sql_ids  = sql_ids.to(device)
            sql_mask = sql_mask.to(device)
            h        = h.to(device)
            y        = y.to(device)
            lbl      = best_labels[q_idx].to(device)

            # forward ---------------------------------------------------
            z = encoder(g, sql_ids, sql_mask)
            z = torch.cat([z, X_pca_t[q_idx]], dim=1)

            mu, sig, logits = head(z, h)

            # losses ----------------------------------------------------
            if ep <= warmup:
                loss_pred = (mu - y).pow(2).mean()
            else:
                loss_pred = GNIMCModel.nll(mu, sig, y)

            # ranking loss
            if B >= 2:
                half = B // 2
                idx  = torch.randperm(B, device=device)
                y1, y2 = y[idx[:half]], y[idx[half : half * 2]]
                m1, m2 = mu[idx[:half]], mu[idx[half : half * 2]]
                mask   = y1 != y2
                if mask.any():
                    tgt = torch.where(y1 < y2,
                                      torch.ones_like(y1),
                                      -torch.ones_like(y1))
                    loss_rank = rank_fn(m1[mask], m2[mask], tgt[mask])
                else:
                    loss_rank = torch.tensor(0.0, device=device)
            else:
                loss_rank = torch.tensor(0.0, device=device)

            loss_ce = ce_fn(logits, lbl)

            loss = loss_pred + LAMBDA_RANK * loss_rank + LAMBDA_CE * loss_ce

            # backward --------------------------------------------------
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 5.0)
            clip_grad_norm_(head.parameters(),    5.0)
            opt.step()

            # stats
            sum_pred += loss_pred.item() * B
            sum_rank += loss_rank.item() * B
            sum_ce   += loss_ce.item()   * B
            sum_tot  += loss.item()      * B

        N = len(ds)
        print(
            f"Epoch {ep:02d} — "
            f"pred={sum_pred/N:.4f}  rank={sum_rank/N:.4f}  "
            f"ce={sum_ce/N:.4f}  total={sum_tot/N:.4f}"
        )

        scheduler.step()
        torch.save(encoder.state_dict(), models_dir / "plan2vec_ckpt.pt")
        torch.save({"model_state_dict": head.state_dict()}, models_dir / "gnimc_ckpt.pt")

    print("Training complete in", round(time.time() - t0, 2), "seconds.")


if __name__ == "__main__":
    print("🚀 Starting train_gnimc_matrix …")
    main()
