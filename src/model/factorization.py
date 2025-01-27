# train.py

def configure_environment():
    """Dynamically set up sys.path to include the project root."""
    import sys
    import os

    # Dynamically determine the project root and add it to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scripts_path = os.path.join(project_root, "scripts")

    # Remove conflicting paths
    if scripts_path in sys.path:
        sys.path.remove(scripts_path)

    # Add project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Call environment setup before importing anything from src
configure_environment()

import numpy as np
import logging
import time

class MaskedALSNonnegative:
    """
    Implements a masked ALS procedure (Algorithm 1 from your LaTeX),
    adapted for Inductive Matrix Completion (IMC) with nonnegative factors.

    The model is:
        P_model(i,j) = x_i^T W H^T y_j

    We incorporate:
      - Masked fill-in of unobserved entries
      - Nonnegative constraints on W and H
      - Feature matrices X and Y
      - Regularization
      - Logging of predicted latencies and (optionally) MSE/MAE on observed entries
    """

    def __init__(
        self,
        k: int,
        reg_param: float = 0.1,
        nonneg: bool = True,
        max_iters: int = 20,
        random_seed: int = 42,
        logger: logging.Logger = None
    ):
        """
        Parameters
        ----------
        k : int
            Latent dimension (rank).
        reg_param : float
            Regularization parameter (lambda).
        nonneg : bool
            If True, clamp W and H to nonnegative after each update.
        max_iters : int
            Maximum number of ALS iterations.
        random_seed : int
            Seed for reproducible random initialization (if needed).
        logger : logging.Logger
            Logger for info/debug. If None, a default will be used.
        """
        self.k = k
        self.reg_param = reg_param
        self.nonneg = nonneg
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Internal factor matrices
        self.W = None  # shape: (F_q x k)
        self.H = None  # shape: (F_h x k)

        # For optional logging
        self.iteration_times = []

    def fit(
        self,
        P_tilde: np.ndarray,   # shape (N_q x N_h)
        M: np.ndarray,         # shape (N_q x N_h), 1=observed, 0=unobserved
        X: np.ndarray,         # shape (N_q x F_q)
        Y: np.ndarray,         # shape (N_h x F_h)
        P_actual: np.ndarray = None
    ) -> np.ndarray:
        """
        Perform masked ALS with nonnegative constraints. Returns the final
        completed matrix P_hat after max_iters iterations.

        Steps each iteration:
          1. Fill unobserved entries of P_tilde with current predicted values
          2. Update W
          3. Fill again
          4. Update H
          5. Log sum of predicted latencies + MSE/MAE on observed if P_actual given
          6. (Optional) compute & log a "loss" that includes regularization

        Returns
        -------
        P_hat : np.ndarray
            The final predicted matrix after ALS, shape (N_q x N_h).
        """
        np.random.seed(self.random_seed)

        N_q, N_h = P_tilde.shape
        F_q = X.shape[1]  # query feature dim
        F_h = Y.shape[1]  # hint feature dim

        # Initialize W, H if not set
        if self.W is None or self.H is None:
            W_init = np.random.randn(F_q, self.k) * 0.01
            H_init = np.random.randn(F_h, self.k) * 0.01
            if self.nonneg:
                W_init = np.abs(W_init)
                H_init = np.abs(H_init)
            self.W = W_init
            self.H = H_init

        # Working copy for partial fill
        P_work = P_tilde.copy()
        self.iteration_times.clear()

        for iter_idx in range(self.max_iters):
            start_time = time.time()

            # (a) fill unobserved
            P_pred_model = self._predict(X, Y)  # shape (N_q x N_h)
            P_work = M * P_work + (1 - M) * P_pred_model

            # (b) update W
            self._update_W(P_work, M, X, Y)

            # (c) fill again
            P_pred_model = self._predict(X, Y)
            P_work = M * P_work + (1 - M) * P_pred_model

            # (d) update H
            self._update_H(P_work, M, X, Y)

            # Build final "mixed" matrix after iteration
            P_hat_iter = M * P_work + (1 - M) * self._predict(X, Y)

            sum_latencies = P_hat_iter.sum()

            if P_actual is not None:
                obs_idx = np.where(M == 1)
                diff = P_actual[obs_idx] - P_hat_iter[obs_idx]
                mse_val = np.mean(diff**2)
                mae_val = np.mean(np.abs(diff))

                self.logger.info(
                    f"[ALS] Iter {iter_idx+1}/{self.max_iters} => "
                    f"Sum of latencies: {sum_latencies:.4f}, "
                    f"MSE_obs: {mse_val:.4f}, MAE_obs: {mae_val:.4f}"
                )
            else:
                self.logger.info(
                    f"[ALS] Iter {iter_idx+1}/{self.max_iters} => "
                    f"Sum of latencies: {sum_latencies:.4f}"
                )

            elapsed = time.time() - start_time
            self.iteration_times.append(elapsed)

        # Final fill
        final_P_pred = self._predict(X, Y)
        P_hat = M * P_work + (1 - M) * final_P_pred
        return P_hat

    def _predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        The model's actual predicted matrix:
        P_pred(i,j) = x_i^T W H^T y_j
        """
        return (X @ self.W) @ ((Y @ self.H).T)

    def _update_W(
        self,
        P_work: np.ndarray,
        M: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray
    ):
        """Update W with H fixed."""
        Z_tilde = Y @ self.H  # (N_h x k)
        E = (M * P_work) @ Z_tilde  # (N_q x k)
        numerator_W = X.T @ E

        ZtZ = Z_tilde.T @ Z_tilde
        denom = ZtZ + self.reg_param * np.eye(self.k)

        try:
            W_new = numerator_W @ np.linalg.inv(denom)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular matrix while updating W. Fallback to pseudo-solution.")
            W_new = numerator_W

        if self.nonneg:
            W_new[W_new < 0] = 0

        self.W = W_new

    def _update_H(
        self,
        P_work: np.ndarray,
        M: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray
    ):
        """Update H with W fixed."""
        Z_tilde2 = X @ self.W  # (N_q x k)
        E2 = (M * P_work).T @ Z_tilde2  # (N_h x k)
        numerator_H = Y.T @ E2

        Z2tZ2 = Z_tilde2.T @ Z_tilde2
        denom2 = Z2tZ2 + self.reg_param * np.eye(self.k)

        try:
            H_new = numerator_H @ np.linalg.inv(denom2)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular matrix while updating H. Fallback to pseudo-solution.")
            H_new = numerator_H

        if self.nonneg:
            H_new[H_new < 0] = 0

        self.H = H_new


class LimeQO:
    """
    Implements your Algorithm 2 (LimeQO):
      - Iteratively reveal new hint-query pairs based on
        delta = min(P_tilde[i,:]) - P_hat_model[i,j].
      - Possibly do fallback random selection.
      - Stop after top_m picks or if no further hints can be selected.
      - Log final sum of latencies + error after the last iteration.
    """

    def __init__(
        self,
        als_solver: MaskedALSNonnegative,
        top_m: int = 5,
        max_iters: int = 10,
        fallback_random: bool = True,
        logger: logging.Logger = None
    ):
        self.als_solver = als_solver
        self.top_m = top_m
        self.max_iters = max_iters
        self.fallback_random = fallback_random
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.iteration_times = []

    def run(
        self,
        P_tilde: np.ndarray,
        M: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        P_actual: np.ndarray = None
    ):
        """
        Perform LimeQO. Each iteration:
          1. Run ALS => get model predictions
          2. For each row i, find argmin_j of P_hat_model[i,j], compute delta_i
          3. Reveal top_m pairs with positive delta
          4. Repeat or stop
        """
        N_q, N_h = P_tilde.shape
        hint_selections = []

        for iteration in range(1, self.max_iters + 1):
            start_time = time.time()

            # Run masked ALS
            P_hat = self.als_solver.fit(
                P_tilde, M, X, Y,
                P_actual=P_actual
            )

            # Now P_hat is the "mixed" matrix, but we get model predictions
            P_hat_model = self.als_solver._predict(X, Y)

            row_min_indices_hat = np.argmin(P_hat_model, axis=1)
            min_tilde_per_row = np.min(P_tilde, axis=1)

            S = []
            for i in range(N_q):
                j = row_min_indices_hat[i]
                delta_ij = min_tilde_per_row[i] - P_hat_model[i, j]
                if delta_ij > 0:
                    S.append((i, j, delta_ij))

            S.sort(key=lambda x: x[2], reverse=True)
            selected_pairs = S[:self.top_m]

            # fallback random if needed
            needed = self.top_m - len(selected_pairs)
            if needed > 0 and self.fallback_random:
                unobs_indices = np.argwhere(M == 0)
                np.random.shuffle(unobs_indices)
                extras = []
                for (ri, ci) in unobs_indices:
                    if not any(sp[0] == ri and sp[1] == ci for sp in selected_pairs):
                        extras.append((ri, ci, 0.0))
                        if len(extras) == needed:
                            break
                selected_pairs += extras

            new_hints = 0
            for (i, j, delta) in selected_pairs:
                if M[i, j] == 0:
                    M[i, j] = 1
                    if P_actual is not None:
                        # reveal ground-truth
                        P_tilde[i, j] = P_actual[i, j]
                    else:
                        # or fill with predicted
                        P_tilde[i, j] = P_hat_model[i, j]
                    hint_selections.append((i, j))
                    new_hints += 1

            elapsed = time.time() - start_time
            self.iteration_times.append(elapsed)
            self.logger.info(
                f"[LimeQO] Iter {iteration}/{self.max_iters}: selected {new_hints} pairs "
                f"({elapsed:.4f} sec)"
            )

            if new_hints == 0:
                self.logger.info("[LimeQO] No further hints can be selected. Stopping.")
                break
            if not np.any(M == 0):
                self.logger.info("[LimeQO] All entries revealed. Stopping.")
                break

        # One final ALS
        P_hat_final = self.als_solver.fit(
            P_tilde, M, X, Y,
            P_actual=P_actual
        )
        P_hat_model_final = self.als_solver._predict(X, Y)
        sum_latencies = P_hat_model_final.sum()

        if P_actual is not None:
            obs_idx = np.where(M == 1)
            if len(obs_idx[0]) > 0:
                diff = P_actual[obs_idx] - P_hat_model_final[obs_idx]
                final_mse = np.mean(diff**2)
                final_mae = np.mean(np.abs(diff))
                self.logger.info(
                    f"[LimeQO] Final sum of latencies: {sum_latencies:.4f}, "
                    f"MSE_observed: {final_mse:.4f}, MAE_observed: {final_mae:.4f}"
                )
            else:
                self.logger.info(
                    f"[LimeQO] Final sum of latencies: {sum_latencies:.4f}. "
                    "No observed entries, so MSE/MAE undefined."
                )
        else:
            self.logger.info(
                f"[LimeQO] Final sum of latencies: {sum_latencies:.4f}"
            )

        return hint_selections, P_hat_final


##############################################################################
#                           EXAMPLE MAIN SCRIPT
##############################################################################

if __name__ == "__main__":
    import argparse
    import logging
    import os
    import numpy as np
    from src.utils.logger import setup_logger
    from src.utils.config_manager import load_config

    parser = argparse.ArgumentParser(description="Train with MaskedALSNonnegative + LimeQO")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("production_limeqo", config["logging"]["log_file"],
                          level=getattr(logging, config["logging"]["log_level"]))

    # Here you would load your real X, Y, P, M from the processed .parquet / .npz files
    # For example:
    #
    # X_df = pd.read_parquet(config["paths"]["processed_features"] + "/X.parquet")
    # Y_df = pd.read_parquet(config["paths"]["processed_features"] + "/Y.parquet")
    # X = X_df.values
    # Y = Y_df.values
    #
    # P_csr = load_npz(config["paths"]["processed_features"] + "/P.npz").tocoo()
    # M_csr = load_npz(config["paths"]["processed_features"] + "/M.npz").tocoo()
    # P = P_csr.toarray()
    # M = M_csr.toarray()
    #
    # P_actual = P.copy()  # if you have the ground truth latencies
    #
    # Then run masked ALS + LimeQO:
    #
    # als_solver = MaskedALSNonnegative(
    #     k=config["training"]["latent_dim"],
    #     reg_param=config["training"]["lambda_reg"],
    #     nonneg=True,
    #     max_iters=config["training"]["max_iters"],
    #     random_seed=42,
    #     logger=logger
    # )
    # limeqo_runner = LimeQO(
    #     als_solver=als_solver,
    #     top_m=config["training"]["top_m"],
    #     max_iters=3,
    #     fallback_random=True,
    #     logger=logger
    # )
    #
    # hint_selections, P_hat_final = limeqo_runner.run(
    #     P_tilde=P.copy(),  # or partial
    #     M=M.copy(),
    #     X=X,
    #     Y=Y,
    #     P_actual=P_actual
    # )
    #
    # logger.info(f"Selected hint-query pairs: {hint_selections}")
    # logger.info(f"Final revealed count: {np.sum(M)} out of {X.shape[0]*Y.shape[0]}")
    #
    # # Evaluate on all revealed entries if desired
    # obs_idx = np.where(M == 1)
    # diff = P_actual[obs_idx] - P_hat_final[obs_idx]
    # mse = np.mean(diff**2)
    # mae = np.mean(np.abs(diff))
    # sum_latencies = P_hat_final.sum()
    # logger.info(
    #     f"FINAL => MSE on revealed: {mse:.4f}, MAE on revealed: {mae:.4f}, "
    #     f"Sum of predicted latencies: {sum_latencies:.4f}"
    # )
    #
    # # Done.
    logger.info("Finished. No synthetic data used.")
