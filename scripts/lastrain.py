#!/usr/bin/env python3
"""
Training Script for Inductive Matrix Completion (IMC) and Residual Model Using 25 Query Features
with Integrated Min-Max Scaling (range 0 to 4).

Outputs:
  - U_best.npy and V_best.npy (IMC factors)
  - residual_model_xgb.pkl, residual_model_poly.pkl, residual_model_scaler.pkl (for cost correction)
  
Ensure that all file paths (for data, training queries, etc.) are correct.
"""
print("Rahim")
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import re
import sqlglot
from sqlglot import exp
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

# ---------------- Environment Configuration ----------------
def configure_environment():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scripts_path = os.path.join(project_root, "scripts")
    if scripts_path in sys.path:
        sys.path.remove(scripts_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

configure_environment()

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------- IMC (GNIMC) Functions ----------------

INIT_WITH_SVD = 0
INIT_WITH_RANDOM = 1

def generate_product_matrix(A, B):
    assert A.shape[0] == B.shape[1], 'Error: dimension mismatch'
    m = A.shape[0]
    M = np.zeros((m, A.shape[1] * B.shape[0]))
    for i in range(m):
        AB = np.outer(A[i, :], B[:, i])
        M[i, :] = AB.flatten()
    return M

def GNIMC(X, omega, rank, A, B, verbose=True, alpha=0.1, perform_qr=True, max_outer_iter=100):
    n1, n2 = X.shape
    d1 = A.shape[1]
    d2 = B.shape[1]
    m = omega.count_nonzero()
    p = m / (n1 * n2)
    I, J, _ = sparse.find(omega)

    # Initial estimate (using SVD)
    from scipy.sparse.linalg import svds
    L, S, R = svds(X / p, k=rank, tol=1e-16)
    U = A.T @ L @ np.diag(np.sqrt(S))
    V = B.T @ R.T @ np.diag(np.sqrt(S))

    AU = A[I, :] @ U
    BV = B[J, :] @ V
    x = X[I, J]
    X_norm = np.linalg.norm(x)
    best_relRes = float("inf")
    U_best = U.copy()
    V_best = V.copy()
    x_hat = np.sum(AU * BV, axis=1)

    for iter_num in range(1, max_outer_iter+1):
        if perform_qr:
            U_Q, U_R = np.linalg.qr(U)
            V_Q, V_R = np.linalg.qr(V)
            AU_q = A[I, :] @ U_Q
            BV_q = B[J, :] @ V_Q
        else:
            AU_q = AU
            BV_q = BV

        L1 = generate_product_matrix(A[I, :], BV_q.T)
        L2 = generate_product_matrix(AU_q, B[J, :].T)
        L_combined = sparse.csr_matrix(np.concatenate((L1, L2), axis=1))
        b = x + alpha * np.sum(AU * BV_q, axis=1)
        z = sp_linalg.lsqr(L_combined, b)[0]
        U_tilde = np.reshape(z[:d1*rank], (d1, rank))
        V_tilde = np.reshape(z[d1*rank:], (rank, d2)).T
        if perform_qr:
            U_tilde = U_tilde @ np.linalg.inv(V_R).T
            V_tilde = V_tilde @ np.linalg.inv(U_R).T
        U = 0.5 * (1 - alpha) * U + U_tilde
        V = 0.5 * (1 - alpha) * V + V_tilde
        AU = A[I, :] @ U
        BV = B[J, :] @ V
        x_hat = np.sum(AU * BV, axis=1)
        relRes = np.linalg.norm(x_hat - x) / X_norm
        if verbose:
            logger.info(f"[GNIMC] Iteration {iter_num}, Relative Residual: {relRes:.6e}")
        if relRes < best_relRes:
            best_relRes = relRes
            U_best = U.copy()
            V_best = V.copy()
    X_hat_final = A @ U_best @ V_best.T @ B.T
    return X_hat_final, U_best, V_best

# ---------------- SQL Feature Extraction (25 features) ----------------
# These functions are taken (and slightly adapted) from your inference code.

def build_alias_mapping(stmt):
    alias_mapping = {}
    for table in stmt.find_all(exp.Table):
        alias = table.alias_or_name
        name = table.name
        alias_mapping[alias] = name
    return alias_mapping

def count_predicates(where_clause):
    if where_clause is None:
        return 0
    predicate_types = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ, exp.Like, exp.ILike, exp.In, exp.Between)
    predicates = list(where_clause.find_all(*predicate_types))
    return len(predicates)

def count_joins(where_clause, alias_mapping):
    if where_clause is None:
        return 0
    join_pairs = set()
    for predicate in where_clause.find_all(exp.EQ):
        left = predicate.left
        right = predicate.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = alias_mapping.get(left.table)
            right_table = alias_mapping.get(right.table)
            if left_table and right_table and left_table != right_table:
                pair = tuple(sorted([left_table, right_table]))
                join_pairs.add(pair)
    return len(join_pairs)

def count_aggregate_functions(select_expressions):
    aggregate_functions = {'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'MEDIAN', 'MODE'}
    count = 0
    for expr in select_expressions:
        if isinstance(expr, exp.Alias):
            expr = expr.this
        if isinstance(expr, exp.Func) and expr.name.upper() in aggregate_functions:
            count += 1
    return count

def count_logical_operators(where_clause):
    if where_clause is None:
        return 0
    logical_ops = (exp.And, exp.Or, exp.Not)
    return sum(1 for _ in where_clause.find_all(*logical_ops))

def count_comparison_operators(where_clause):
    if where_clause is None:
        return 0
    comparison_ops = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ)
    return sum(1 for _ in where_clause.find_all(*comparison_ops))

def count_group_by_columns(stmt):
    group = stmt.find(exp.Group)
    if group:
        return len(list(group.find_all(exp.Column)))
    return 0

def count_order_by_columns(stmt):
    order = stmt.find(exp.Order)
    if order:
        return len(list(order.find_all(exp.Ordered)))
    return 0

def count_nested_subqueries(stmt):
    subqueries = list(stmt.find_all(exp.Select))
    return max(0, len(subqueries) - 1)

def count_correlated_subqueries(stmt):
    correlated = False
    for subquery in stmt.find_all(exp.Select):
        if subquery is stmt:
            continue
        outer_tables = {t.alias_or_name for t in stmt.find_all(exp.Table)}
        for col in subquery.find_all(exp.Column):
            if col.table and col.table in outer_tables:
                correlated = True
                break
        if correlated:
            break
    return 1.0 if correlated else 0.0

def count_case_statements(stmt):
    return len(list(stmt.find_all(exp.Case)))

def count_union_operations(stmt):
    return len(list(stmt.find_all(exp.Union)))

def parse_sql(query_str):
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        stmt = statements[0]
        alias_mapping = build_alias_mapping(stmt)
        tables = [t.alias_or_name for t in stmt.find_all(exp.Table)]
        num_tables = len(tables)
        where_clause = stmt.find(exp.Where)
        num_predicates = count_predicates(where_clause)
        num_joins = count_joins(where_clause, alias_mapping)
        has_group_by = 1.0 if stmt.find(exp.Group) else 0.0
        has_order_by = 1.0 if stmt.find(exp.Order) else 0.0
        num_nested_subqueries = count_nested_subqueries(stmt)
        has_subquery = 1.0 if num_nested_subqueries > 0 else 0.0
        query_length = len(query_str)
        num_tokens = len(list(stmt.walk()))
        select = stmt.find(exp.Select)
        select_expressions = select.expressions if select else []
        num_select_expressions = len(select_expressions)
        has_distinct = 1.0 if select and select.args.get("distinct") else 0.0
        num_aggregate_functions = count_aggregate_functions(select_expressions)
        num_logical_operators = count_logical_operators(where_clause)
        num_comparison_operators = count_comparison_operators(where_clause)
        num_group_by_columns = count_group_by_columns(stmt)
        num_order_by_columns = count_order_by_columns(stmt)
        has_correlated_subqueries = count_correlated_subqueries(stmt)
        num_inner_joins = len([j for j in stmt.find_all(exp.Join) if j.args.get('kind') and j.args.get('kind').upper()=='INNER'])
        num_left_joins = len([j for j in stmt.find_all(exp.Join) if j.args.get('kind') and j.args.get('kind').upper()=='LEFT'])
        num_right_joins = len([j for j in stmt.find_all(exp.Join) if j.args.get('kind') and j.args.get('kind').upper()=='RIGHT'])
        num_full_outer_joins = len([j for j in stmt.find_all(exp.Join) if j.args.get('kind') and j.args.get('kind').upper() in ['FULL OUTER','FULLOUTER','FULL_OUTER']])
        has_limit = 1.0 if stmt.find(exp.Limit) else 0.0
        has_union = 1.0 if stmt.find(exp.Union) else 0.0
        num_union_operations = count_union_operations(stmt)
        num_case_statements = count_case_statements(stmt)
        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery,
            "query_length": query_length,
            "num_tokens": num_tokens,
            "num_select_expressions": num_select_expressions,
            "has_distinct": has_distinct,
            "num_aggregate_functions": num_aggregate_functions,
            "num_logical_operators": num_logical_operators,
            "num_comparison_operators": num_comparison_operators,
            "num_group_by_columns": num_group_by_columns,
            "num_order_by_columns": num_order_by_columns,
            "num_nested_subqueries": num_nested_subqueries,
            "has_correlated_subqueries": has_correlated_subqueries,
            "num_inner_joins": num_inner_joins,
            "num_left_joins": num_left_joins,
            "num_right_joins": num_right_joins,
            "num_full_outer_joins": num_full_outer_joins,
            "has_limit": has_limit,
            "has_union": has_union,
            "num_union_operations": num_union_operations,
            "num_case_statements": num_case_statements
        }
        return features
    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {key: 0.0 for key in [
            "num_tables", "num_joins", "num_predicates", "has_group_by", "has_order_by",
            "has_subquery", "query_length", "num_tokens", "num_select_expressions", "has_distinct",
            "num_aggregate_functions", "num_logical_operators", "num_comparison_operators",
            "num_group_by_columns", "num_order_by_columns", "num_nested_subqueries",
            "has_correlated_subqueries", "num_inner_joins", "num_left_joins", "num_right_joins",
            "num_full_outer_joins", "has_limit", "has_union", "num_union_operations", "num_case_statements"
        ]}

def prepare_25_features(feats_dict):
    feature_order = [
        "num_tables",
        "num_joins",
        "num_predicates",
        "has_group_by",
        "has_order_by",
        "has_subquery",
        "query_length",
        "num_tokens",
        "num_select_expressions",
        "has_distinct",
        "num_aggregate_functions",
        "num_logical_operators",
        "num_comparison_operators",
        "num_group_by_columns",
        "num_order_by_columns",
        "num_nested_subqueries",
        "has_correlated_subqueries",
        "num_inner_joins",
        "num_left_joins",
        "num_right_joins",
        "num_full_outer_joins",
        "has_limit",
        "has_union",
        "num_union_operations",
        "num_case_statements"
    ]
    raw_features = [feats_dict.get(key, 0.0) for key in feature_order]
    return np.array(raw_features, dtype=np.float32).reshape(1, -1)

# ---------------- Residual Model Training ----------------
# For each training query we assume a measured "true latency" (loaded from CSV)
# and we use the IMC model to compute an initial cost prediction.
# Then the residual = (true latency - initial cost) is learned.
# We form training examples by concatenating the scaled 25-feature vector and the best hint vector.
# (Hint matrix Y is built from COMBO_STRS and HINTS defined below.)

HINTS = [
    "enable_hashjoin",
    "enable_indexonlyscan",
    "enable_indexscan",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_seqscan"
]

COMBO_STRS = [
    "hashjoin,indexonlyscan",
    "hashjoin,indexonlyscan,indexscan",
    "hashjoin,indexonlyscan,indexscan,mergejoin",
    "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop",
    "hashjoin,indexonlyscan,indexscan,mergejoin,seqscan",
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
    "hashjoin,indexscan",
    "hashjoin,indexscan,mergejoin",
    "hashjoin,indexscan,mergejoin,nestloop",
    "hashjoin,indexscan,mergejoin,nestloop,seqscan",
    "hashjoin,indexscan,mergejoin,seqscan",
    "hashjoin,indexscan,nestloop",
    "hashjoin,indexscan,nestloop,seqscan",
    "hashjoin,indexscan,seqscan",
    "hashjoin,mergejoin,nestloop,seqscan",
    "hashjoin,mergejoin,seqscan",
    "hashjoin,nestloop,seqscan",
    "hashjoin,seqscan",
    "indexonlyscan,indexscan,mergejoin",
    "indexonlyscan,indexscan,mergejoin,nestloop",
    "indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
    "indexonlyscan,indexscan,mergejoin,seqscan",
    "indexonlyscan,indexscan,nestloop",
    "indexonlyscan,indexscan,nestloop,seqscan",
    "indexonlyscan,mergejoin",
    "indexonlyscan,mergejoin,nestloop",
    "indexonlyscan,mergejoin,nestloop,seqscan",
    "indexonlyscan,mergejoin,seqscan",
    "indexonlyscan,nestloop",
    "indexonlyscan,nestloop,seqscan",
    "indexscan,mergejoin",
    "indexscan,mergejoin,nestloop",
    "indexscan,mergejoin,nestloop,seqscan",
    "indexscan,mergejoin,seqscan",
    "indexscan,nestloop",
    "indexscan,nestloop,seqscan",
    "mergejoin,nestloop,seqscan",
    "mergejoin,seqscan",
    "nestloop,seqscan"
]

def build_hint_matrix(combo_strs, hint_list):
    N = len(combo_strs)
    D = len(hint_list)
    Y = np.zeros((N, D), dtype=float)
    for i, c_str in enumerate(combo_strs):
        items = [item.strip() for item in c_str.split(",") if item.strip()]
        for item in items:
            enable_str = "enable_" + item
            if enable_str in hint_list:
                j = hint_list.index(enable_str)
                Y[i, j] = 1.0
    return Y

def train_residual_model(X_train_features, residuals_train):
    """
    Train an XGBoost regressor on the concatenated features.
    
    X_train_features: numpy.ndarray of shape (N, n_features_total)
    residuals_train: numpy.ndarray of shape (N,)
    
    Returns:
      best_model, poly_transformer
    """
    # Set up initial regressor and parameter grid
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [500, 1000]
    }
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    # First, apply a polynomial expansion to the features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_train_features)
    grid_search.fit(X_poly, residuals_train)
    best_model = grid_search.best_estimator_
    # Refit best_model on all data with early stopping (using a 20% validation split)
    X_tr, X_val, y_tr, y_val = train_test_split(X_poly, residuals_train, test_size=0.2, random_state=42)
    best_model.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   early_stopping_rounds=50,
                   verbose=False)
    return best_model, poly



def main():
    # Paths (adjust as needed)
    MODEL_DIR = "/Users/raahimlone/New_Data"
    TRAIN_QUERIES_DIR = "/Users/raahimlone/rahhh/Data_Gathering/training_sql_queries"
    LATENCIES_CSV = "/Users/raahimlone/rahhh/training_latencies.csv"  
    # Load IMC training data
    try:
        W_true = np.load(os.path.join(MODEL_DIR, "W_low_rank.npy"))
        M_mask = np.load(os.path.join(MODEL_DIR, "M.npy"))
        data_X = np.load(os.path.join(MODEL_DIR, "X_scaled.npz"))
        A = data_X['data']
        data_Y = np.load(os.path.join(MODEL_DIR, "Y_scaled.npz"))
        B = data_Y['Y']
        logger.info("Loaded W_true, M, A, and B for IMC training.")
    except Exception as e:
        logger.error(f"Error loading IMC training data: {e}")
        return

    # Run GNIMC to obtain IMC factors
    omega = sparse.csr_matrix(M_mask)
    X_hat_imc, U_best, V_best = GNIMC(W_true, omega, rank=5, A=A, B=B, verbose=True)
    np.save(os.path.join(MODEL_DIR, "U_best.npy"), U_best)
    np.save(os.path.join(MODEL_DIR, "V_best.npy"), V_best)
    logger.info("IMC training complete. U_best and V_best saved.")

    # Build IMC prediction matrix: Z = U_best @ V_best.T
    Z = U_best @ V_best.T

    # Build hint matrix Y for all combinations
    Y_hint = build_hint_matrix(COMBO_STRS, HINTS)

    # Load training query true latencies
    try:
        df_latencies = pd.read_csv(LATENCIES_CSV)  # expect columns: filename, latency
    except Exception as e:
        logger.error(f"Error loading training latencies CSV: {e}")
        return

    # For each training query, extract 25 features and compute the IMC initial cost.
    query_files = [f for f in os.listdir(TRAIN_QUERIES_DIR) if f.endswith(".sql")]
    X_queries = []  # to hold raw 25 features
    best_hint_vectors = []  # best Y vector for each query (from IMC predictions)
    true_costs = []  # measured latency
    initial_costs = []  # predicted initial cost for the chosen best combo

    for fname in query_files:
        full_path = os.path.join(TRAIN_QUERIES_DIR, fname)
        try:
            with open(full_path, 'r', encoding="utf-8") as f:
                query_str = f.read().strip()
            if not query_str:
                logger.warning(f"{fname} is empty. Skipping.")
                continue
            feats = parse_sql(query_str)
            raw_feat = prepare_25_features(feats)  # shape (1,25)
            X_queries.append(raw_feat)
            # Use the IMC model: initial cost = (raw_feat @ Z) @ Y_hint.T.
            # (Note: here raw_feat dimensions should match Z; if not, assume that in training
            # the side-information for queries is represented by the 25 features.)
            # In this example, we use the same 25 features.
            init_costs = raw_feat @ Z  # shape (1, number_of_hints) assuming Z is sized (25,6)
            # To mimic inference, we compute predicted cost for each hint combo:
            init_costs_combo = init_costs @ Y_hint.T  # shape (1, number_of_combos)
            init_costs_combo = init_costs_combo.flatten()
            best_idx = np.argmin(init_costs_combo)
            best_hint_vector = Y_hint[best_idx]  # shape (6,)
            best_hint_vectors.append(best_hint_vector.reshape(1, -1))
            initial_costs.append(init_costs_combo[best_idx])
            # Look up true latency for this query from the CSV (match by filename)
            row = df_latencies[df_latencies['filename'] == fname]
            if row.empty:
                logger.warning(f"No latency found for {fname}. Skipping.")
                continue
            true_latency = float(row.iloc[0]['latency'])
            true_costs.append(true_latency)
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")

    if not X_queries:
        logger.error("No training queries processed successfully. Exiting.")
        return

    # Stack features: X_queries becomes (N,25) and best_hint_vectors becomes (N,6)
    X_queries_all = np.vstack(X_queries)  # (N,25)
    Y_best_all = np.vstack(best_hint_vectors)  # (N,6)

    # We now train a MinMaxScaler on the 25 features (range 0 to 4)
    scaler = MinMaxScaler(feature_range=(0,4))
    X_queries_scaled = scaler.fit_transform(X_queries_all)
    # Save the scaler for later use in inference.
    joblib.dump(scaler, os.path.join(MODEL_DIR, "residual_model_scaler.pkl"))
    logger.info("MinMaxScaler fitted and saved.")

    # Compute residuals: target = true_latency - initial_cost
    residuals = np.array(true_costs) - np.array(initial_costs)  # shape (N,)

    # Prepare training examples for residual model: concatenate scaled query features and best hint vector.
    X_residual = np.hstack((X_queries_scaled, Y_best_all))  # shape (N, 25+6 = 31)

    # Train the residual model (with polynomial expansion)
    best_model, poly = train_residual_model(X_residual, residuals)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "residual_model_xgb.pkl"))
    joblib.dump(poly, os.path.join(MODEL_DIR, "residual_model_poly.pkl"))
    logger.info("Residual model and polynomial transformer trained and saved.")

if __name__ == "__main__":
    main()
