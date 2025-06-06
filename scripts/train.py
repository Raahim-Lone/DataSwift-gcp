#!/usr/bin/env python3
"""
Gauss–Newton Inductive Matrix Completion (GNIMC) + LIME‑QO + Residual XGB
──────────────────────────────────────────────────────────────────────────
**This revision targets small‑rank (r = 5) IMDB‑sized latency matrices.**
Key improvements over the previous script
────────────────────────────────────────
1.   **Numerical stability** – side‑information is whitened once and reused.
2.   **Tikhonov damping** – GNIMC now honours `lambda_` through LSQR’s `damp`.
3.   **Faster convergence** – replace 0.48 shrinkage with a tunable step‑size γ.
4.   **Residual model** – XGBoost is regularised and polynomial blow‑up removed.
5.   **Evaluation helpers** – RMSE on unobserved cells reported automatically.

Author: Pini Zilber & Boaz Nadler (orig.) | Revision: ChatGPT (o3) 2025‑04‑22
"""

from __future__ import annotations
import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List

from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# ────────────────────────────────────────────────────────────────────────────
# 1.  Small helpers
# ────────────────────────────────────────────────────────────────────────────

def whiten(matrix: np.ndarray, name: str) -> Tuple[np.ndarray, StandardScaler]:
    """Return z‑scored copy and fitted scaler."""
    scaler = StandardScaler(with_mean=True)
    mat_w = scaler.fit_transform(matrix)
    print(f"✅ {name} whitened: mean≈{mat_w.mean():.2e}, std≈{mat_w.std():.2e}")
    return mat_w, scaler


def generate_product_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return M such that M @ vec(C) = vec(A @ C @ B)."""
    assert A.shape[0] == B.shape[1], "Dimension mismatch in generate_product_matrix"
    m = A.shape[0]
    M = np.empty((m, A.shape[1] * B.shape[0]))
    for i in range(m):
        M[i, :] = np.outer(A[i, :], B[:, i]).ravel()
    return M


# ────────────────────────────────────────────────────────────────────────────
# 2.  GNIMC core (rank fixed to 5)
# ────────────────────────────────────────────────────────────────────────────

INIT_WITH_SVD, INIT_WITH_RANDOM, INIT_WITH_USER_DEFINED = range(3)

def GNIMC(
    X: np.ndarray,
    omega: sparse.csr_matrix,
    rank: int,
    A: np.ndarray,
    B: np.ndarray,
    *,
    lambda_: float = 1e-2,
    gamma: float = 0.2,
    max_outer_iter: int = 100,
    inner_iter_init: int = 30,
    tol: float = 1e-12,
    init_option: int = INIT_WITH_SVD,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, bool, List[float], np.ndarray, np.ndarray]:
    """Minimal but robust GNIMC implementation with damping and step‑size γ."""
    n1, n2 = X.shape
    I, J, _ = sparse.find(omega)

    # ── Init U, V ───────────────────────────────────────────────────────────
    if init_option == INIT_WITH_SVD:
        p = omega.count_nonzero() / (n1 * n2)
        L, S, R = sp_linalg.svds(X / max(p, 1e-6), k=rank, tol=1e-10)
        U = A.T @ L @ np.diag(np.sqrt(S))
        V = B.T @ R.T @ np.diag(np.sqrt(S))
    elif init_option == INIT_WITH_RANDOM:
        U, _ = np.linalg.qr(np.random.randn(A.shape[1], rank))
        V, _ = np.linalg.qr(np.random.randn(B.shape[1], rank))
    else:
        raise ValueError("User‑defined init not implemented in this revision")

    # ── Pre‑compute stats ───────────────────────────────────────────────────
    x_obs = X[I, J]
    x_norm = np.linalg.norm(x_obs)
    rel_hist: List[float] = []
    best_rel = np.inf
    U_best, V_best = U.copy(), V.copy()

    for t in range(1, max_outer_iter + 1):
        AU = A[I] @ U         # (m, r)
        BV = B[J] @ V         # (m, r)
        x_hat = np.sum(AU * BV, axis=1)

        # ── Build LSQR system  (Gauss–Newton linearisation) ────────────────
        L1 = generate_product_matrix(A[I], BV.T)
        L2 = generate_product_matrix(AU, B[J].T)
        L = sparse.csr_matrix(np.hstack([L1, L2]))
        b = x_obs - x_hat

        z = sp_linalg.lsqr(
            L, b, atol=tol, btol=tol, iter_lim=inner_iter_init, damp=np.sqrt(lambda_)
        )[0]

        U_tilde = z[: A.shape[1] * rank].reshape(A.shape[1], rank)
        V_tilde = z[A.shape[1] * rank :].reshape(rank, B.shape[1]).T

        # ── Update with step‑size γ ────────────────────────────────────────
        U = (1 - gamma) * U + gamma * U_tilde
        V = (1 - gamma) * V + gamma * V_tilde

        # ── Convergence metrics ────────────────────────────────────────────
        rel = np.linalg.norm(b) / x_norm
        rel_hist.append(rel)
        if rel < best_rel:
            best_rel = rel
            U_best, V_best = U.copy(), V.copy()
        if verbose:
            print(f"[GNIMC] iter {t:3d}  relRes={rel:9.3e}")
        if rel < 1e-6:
            break

    X_hat = A @ U_best @ V_best.T @ B.T
    converged = best_rel < 1e-6
    return X_hat, t, converged, rel_hist, U_best, V_best


# ────────────────────────────────────────────────────────────────────────────
# 3.  Residual‑model utilities
# ────────────────────────────────────────────────────────────────────────────

def train_residual_model(
    W_true: np.ndarray,
    X_hat: np.ndarray,
    *,
    A_poly: np.ndarray,
    B_poly: np.ndarray,
    M: np.ndarray,
    model: xgb.XGBRegressor | None = None,
    scaler: StandardScaler | None = None,
):
    """Fit or update XGBoost on observed residuals and return corrected X_hat."""
    resid = (W_true - X_hat).astype(float)
    rows, cols = np.where(M == 1)
    feats = np.hstack([A_poly[rows], B_poly[cols]])

    if scaler is None:
        scaler = StandardScaler().fit(feats)
    feats_s = scaler.transform(feats)
    target = resid[rows, cols]

    if model is None:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=0.2,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(feats_s, target)
    else:
        model.fit(feats_s, target, xgb_model=model)  # warm‑start

    # predict all residuals
    all_feats_s = scaler.transform(np.hstack([
        np.repeat(A_poly, B_poly.shape[0], axis=0),
        np.tile(B_poly, (A_poly.shape[0], 1))
    ]))
    pred = model.predict(all_feats_s).reshape(W_true.shape)
    X_hat_adj = np.maximum(X_hat + pred, 0)
    return X_hat_adj, model, scaler


# ────────────────────────────────────────────────────────────────────────────
# 4.  LIME‑QO (simplified: single pass per alteration)
# ────────────────────────────────────────────────────────────────────────────

def LIMEQO(
    *,
    W_true: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    rank: int,
    lambda_: float,
    m_select: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_q, n_h = W_true.shape
    omega = sparse.csr_matrix(M)
    X_hat, _, _, _, U_best, V_best = GNIMC(
        X=W_true * M,  # zero‑fill unobserved
        omega=omega,
        rank=rank,
        A=A,
        B=B,
        lambda_=lambda_,
        verbose=False,
    )

    # choose best hint per query among **unobserved** entries
    S: List[Tuple[int, int]] = []
    for i in range(n_q):
        unobs = np.where(M[i] == 0)[0]
        if unobs.size == 0:
            continue
        j_star = unobs[np.argmin(X_hat[i, unobs])]
        S.append((i, j_star))
    # pick top‑m_select by improvement (proxy = predicted latency)
    S = sorted(S, key=lambda t: X_hat[t[0], t[1]])[: m_select]

    for i, j in S:
        M[i, j] = 1
    return S, M, X_hat, U_best, V_best


# ────────────────────────────────────────────────────────────────────────────
# 5.  Main driver
# ────────────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path("/Users/raahimlone/New_Data")
    W_true = np.load(data_dir / "W_low_rank.npy")
    M = np.load(data_dir / "M.npy")

    # side information
    A_raw = np.load(data_dir / "X_scaled.npz")["data"]
    B_raw = np.load(data_dir / "Y_scaled.npz")["Y"]

    # ── Whitening once ──────────────────────────────────────────────────────
    A, scaler_A = whiten(A_raw, "A")
    B, scaler_B = whiten(B_raw, "B")

    # reuse whitened features for residual model without polynomial blow‑up
    A_poly, _ = whiten(A, "A_poly")  # simple z‑score acts like identity here
    B_poly, _ = whiten(B, "B_poly")

    rank = 5                     # fixed by user
    lambda_ = 1e-2
    m_select = 5

    # ── Initial GNIMC fit ──────────────────────────────────────────────────
    omega0 = sparse.csr_matrix(M)
    X_hat, _, _, _, U_best, V_best = GNIMC(
        X=W_true * M,
        omega=omega0,
        rank=rank,
        A=A,
        B=B,
        lambda_=lambda_,
        verbose=True,
    )

    # ── Residual correction ────────────────────────────────────────────────
    X_hat, resid_model, resid_scaler = train_residual_model(
        W_true,
        X_hat,
        A_poly=A_poly,
        B_poly=B_poly,
        M=M,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    test_idx = (M == 0)
    rmse = np.sqrt(np.mean((W_true[test_idx] - X_hat[test_idx]) ** 2))
    print(f"🏁  Test RMSE on unobserved cells: {rmse:.4f}\n")

    # ── Save artefacts ─────────────────────────────────────────────────────
    np.save(data_dir / "X_hat_final.npy", X_hat)
    joblib.dump(resid_model, data_dir / "residual_model_xgb.pkl")
    print("✅ Artefacts saved to", data_dir)


if __name__ == "__main__":
    main()
