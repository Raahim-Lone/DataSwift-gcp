#!/usr/bin/env python3
"""
Full Inference Script Using 25 Features for Inductive Matrix Completion Model
with Consistent Standard Scaling & Residual Modeling
### Updated to load the same StandardScaler from training ###
"""

import os
import sys
import logging
import psycopg2
import numpy as np
import pandas as pd
import re
import sqlglot
from sqlglot import exp
import joblib

from sklearn.preprocessing import StandardScaler  # <-- Use the SAME scaler from training

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

# ---------------- Define Hints and Combination Strings ----------------

HINTS = [
    "enable_hashjoin",
    "enable_indexonlyscan",
    "enable_indexscan",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_seqscan"
]

COMBO_STRS = [
    # same combos as your original script ...
    "hashjoin,indexonlyscan",
    "hashjoin,indexonlyscan,indexscan",
    # etc...
    "nestloop,seqscan"
]

# ---------------- Original 25-Feature SQL Parsing Functions ----------------

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
    predicate_types = (
        exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE,
        exp.NEQ, exp.Like, exp.ILike, exp.In, exp.Between
    )
    predicates = list(where_clause.find_all(*predicate_types))
    return len(predicates)

def count_joins(where_clause, alias_mapping):
    if where_clause is None:
        return 0
    join_pairs = set()
    join_predicate_types = (exp.EQ,)
    for predicate in where_clause.find_all(*join_predicate_types):
        left = predicate.left
        right = predicate.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table_alias = left.table
            right_table_alias = right.table
            left_table_name = alias_mapping.get(left_table_alias)
            right_table_name = alias_mapping.get(right_table_alias)
            if left_table_name and right_table_name and left_table_name != right_table_name:
                pair = tuple(sorted([left_table_name, right_table_name]))
                join_pairs.add(pair)
    return len(join_pairs)

def count_aggregate_functions(select_expressions):
    aggregate_functions = {'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'MEDIAN', 'MODE'}
    count = 0
    for expr in select_expressions:
        node = expr.this if isinstance(expr, exp.Alias) else expr
        if isinstance(node, exp.Func) and node.name.upper() in aggregate_functions:
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
        outer_tables = {table.alias_or_name for table in stmt.find_all(exp.Table)}
        for column in subquery.find_all(exp.Column):
            if column.table and column.table in outer_tables:
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
        tables = [node.alias_or_name for node in stmt.find_all(exp.Table)]
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
        list_of_joins = list(stmt.find_all(exp.Join))
        num_inner_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'INNER'])
        num_left_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'LEFT'])
        num_right_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'RIGHT'])
        num_full_outer_joins = len([j for j in list_of_joins
                                    if j.args.get('kind') and j.args.get('kind').upper() in ['FULL OUTER','FULLOUTER','FULL_OUTER']])
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
        return {
            "num_tables": 0, "num_joins": 0, "num_predicates": 0,
            "has_group_by": 0.0, "has_order_by": 0.0, "has_subquery": 0.0,
            "query_length": 0, "num_tokens": 0, "num_select_expressions": 0,
            "has_distinct": 0.0, "num_aggregate_functions": 0, 
            "num_logical_operators": 0, "num_comparison_operators": 0,
            "num_group_by_columns": 0, "num_order_by_columns": 0,
            "num_nested_subqueries": 0, "has_correlated_subqueries": 0.0,
            "num_inner_joins": 0, "num_left_joins": 0, "num_right_joins": 0,
            "num_full_outer_joins": 0, "has_limit": 0.0, "has_union": 0.0,
            "num_union_operations": 0, "num_case_statements": 0
        }

def prepare_features(feats_dict):
    """
    Convert the feature dictionary into a fixed-order numpy array of shape (1, 25).
    """
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
    raw_features = [feats_dict.get(k, 0.0) for k in feature_order]
    return np.array(raw_features, dtype=np.float32).reshape(1, -1)

# ---------------- Hint Matrix Construction ----------------

def build_hint_matrix_from_combos(combo_strs, hint_list):
    N = len(combo_strs)
    D = len(hint_list)
    Y = np.zeros((N, D), dtype=float)
    for i, c_str in enumerate(combo_strs):
        items = [x.strip() for x in c_str.split(",") if x.strip()]
        for item in items:
            enable_str = "enable_" + item
            if enable_str in hint_list:
                j = hint_list.index(enable_str)
                Y[i, j] = 1.0
    return Y

# ---------------- Inductive Matrix Completion Model Class ----------------

class InductiveMatrixCompletionModel:
    """
    IMC model plus residual correction.
    """
    def __init__(self, W, H, residual_model, poly, scaler):
        """
        ### FIX/UPDATE ###
        We store 'scaler' here so we can use it in predict_all_combos.
        """
        self.Z = W @ H.T
        self.residual_model = residual_model
        self.poly = poly
        self.scaler = scaler  # <-- Store loaded scaler

    def predict_all_combos(self, x_feat, Y):
        """
        Predict costs for all hint combos for a set of queries.
        """
        # IMC base predictions
        # x_feat shape: (N, d1)
        # self.Z shape: (d1, d2) if W is (d1, r) and H is (d2, r)
        xZ = x_feat @ self.Z  # shape: (N, d2)
        initial_costs = xZ @ Y.T  # shape: (N, M)  # M combos

        # Prepare features for residual prediction
        num_combos = Y.shape[0]
        N = x_feat.shape[0]
        x_feat_repeated = np.repeat(x_feat, num_combos, axis=0)
        Y_expanded = np.tile(Y, (N, 1))
        features = np.hstack((x_feat_repeated, Y_expanded))  # (N*M, d1 + d2)

        # Polynomial transform
        features_poly = self.poly.transform(features)

        # ### FIX/UPDATE ###
        # Scale polynomial features with the *same* scaler from training
        features_poly_scaled = self.scaler.transform(features_poly)

        # Predict residuals
        residuals_pred = self.residual_model.predict(features_poly_scaled)  # shape: (N*M,)
        adjusted_costs = initial_costs.flatten() + residuals_pred  # shape: (N*M,)
        return adjusted_costs

# ---------------- PostgreSQL Hint Application ----------------

def apply_plan_hints(cursor, plan_vector):
    for i, hint_name in enumerate(HINTS):
        val = plan_vector[i]
        cmd = f"SET {hint_name} TO {'ON' if val >= 0.5 else 'OFF'};"
        cursor.execute(cmd)

def reset_all_hints(cursor):
    cursor.execute("RESET ALL;")

# ---------------- EXPLAIN ANALYZE Parsing ----------------

def parse_explain_analyze_output(explain_output):
    total_time = None
    for row in explain_output:
        line = row[0]
        match_ms = re.search(r"Execution Time:\s+([\d.]+)\s+ms", line)
        if match_ms:
            total_time_ms = float(match_ms.group(1))
            total_time = total_time_ms / 1000.0
            break
    if total_time is None:
        for row in explain_output:
            line = row[0]
            match_s = re.search(r"Execution Time:\s+([\d.]+)\s+s", line)
            if match_s:
                total_time = float(match_s.group(1))
                break
    if total_time is None:
        logger.warning("Could not parse Execution Time from EXPLAIN ANALYZE output.")
        return float("inf")
    return total_time

def run_query_postgres_once_explain_analyze(query_str, plan_vector, pg_host, pg_db, pg_user, pg_password, port=5432):
    conn = None
    try:
        conn = psycopg2.connect(
            host=pg_host, dbname=pg_db, user=pg_user,
            password=pg_password, port=port
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            apply_plan_hints(cur, plan_vector)
            cur.execute(f"EXPLAIN ANALYZE {query_str}")
            explain_output = cur.fetchall()
            reset_all_hints(cur)
            return parse_explain_analyze_output(explain_output)
    except Exception as e:
        logger.error(f"Error executing EXPLAIN ANALYZE: {e}")
        return float("inf")
    finally:
        if conn:
            conn.close()

# ---------------- Main Inference Function ----------------

def main():
    # Configuration
    MODEL_DIR = "/Users/raahimlone/New_Data"
    QUERIES_DIR = "/Users/raahimlone/rahhh/Data_Gathering/raw_sql_queries"
    OUTPUT_CSV = "/Users/raahimlone/rahhh/results_imc_25feat.csv"
    PG_HOST = "localhost"
    PG_DB = "IMDB"
    PG_USER = "postgres"
    PG_PASSWORD = "raahimhere"
    PG_PORT = 6543

    # Load IMC model components
    try:
        W = np.load(os.path.join(MODEL_DIR, "U_best.npy"))
        H = np.load(os.path.join(MODEL_DIR, "V_best.npy"))
        logger.info("✅ Loaded U_best, V_best from training.")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return

    # Load residual model, polynomial, and *scaler*
    try:
        residual_model = joblib.load(os.path.join(MODEL_DIR, "residual_model_xgb.pkl"))
        poly           = joblib.load(os.path.join(MODEL_DIR, "residual_model_poly.pkl"))
        scaler         = joblib.load(os.path.join(MODEL_DIR, "residual_model_scaler.pkl"))
        logger.info("✅ Residual model, polynomial transformer, and scaler loaded.")
    except FileNotFoundError:
        logger.error("Residual model or polynomial/scaler not found.")
        return
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        return

    # Create IMC model with residual
    try:
        imc_model = InductiveMatrixCompletionModel(W, H, residual_model, poly, scaler)
        logger.info("✅ IMC model with residuals initialized.")
    except Exception as e:
        logger.error(f"Error initializing IMC model: {e}")
        return

    # Build hint matrix
    Y = build_hint_matrix_from_combos(COMBO_STRS, HINTS)
    logger.info(f"Hint matrix built with shape {Y.shape}.")

    # Feature collection from queries
    logger.info(f"Extracting features from {QUERIES_DIR} ...")
    all_features = []
    filenames = []
    for fname in os.listdir(QUERIES_DIR):
        if not fname.endswith(".sql"):
            continue
        full_path = os.path.join(QUERIES_DIR, fname)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                query_str = f.read().strip()
            if not query_str:
                logger.warning(f"{fname} is empty.")
                continue

            feats_dict = parse_sql(query_str)
            all_features.append(feats_dict)
            filenames.append(fname)
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")

    if not all_features:
        logger.warning("No query features extracted.")
        print("No results to display.")
        return

    # Convert extracted features to DataFrame
    df_features = pd.DataFrame(all_features).fillna(0)
    logger.info(f"Extracted features for {df_features.shape[0]} queries and {df_features.shape[1]} columns.")

    # Convert DataFrame to numpy array
    X_raw = df_features.to_numpy(dtype=np.float32)  # shape: (N, 25)
    # IMPORTANT: We do NOT fit a new scaler here. We'll just pass these to the model.

    # For each query, we want to predict the best plan among combos
    # predict_all_combos expects shape: (N, d1), so X_raw is that shape if d1 = 25
    # Then it returns shape (N*M, ) costs
    logger.info("Computing predicted costs for all hint combos.")
    adjusted_costs = imc_model.predict_all_combos(X_raw, Y)  # shape: (N*M,)

    N = X_raw.shape[0]
    M = Y.shape[0]
    adjusted_costs_reshaped = adjusted_costs.reshape(N, M)
    best_indices = np.argmin(adjusted_costs_reshaped, axis=1)
    best_combos  = [COMBO_STRS[idx] for idx in best_indices]
    best_Y_vecs  = Y[best_indices]

    # Execute queries with best combos
    logger.info("Executing queries with selected hints.")
    results = []
    for i, fname in enumerate(filenames):
        full_path = os.path.join(QUERIES_DIR, fname)
        query_str = ""
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                query_str = f.read().strip()
            if not query_str:
                logger.warning(f"{fname} is empty.")
                continue

            best_idx = best_indices[i]
            best_combo = best_combos[i]
            plan_vector = best_Y_vecs[i]
            latency = run_query_postgres_once_explain_analyze(
                query_str, plan_vector, PG_HOST, PG_DB, PG_USER, PG_PASSWORD, PG_PORT
            )
            enabled_hints = [HINTS[j].replace("enable_","") for j in range(len(HINTS)) if plan_vector[j] >= 0.5]
            predicted_cost = adjusted_costs_reshaped[i, best_idx]

            results.append({
                "filename": fname,
                "best_idx": best_idx,
                "best_combo": best_combo,
                "predicted_cost": predicted_cost,
                "latency": latency,
                "hints": ",".join(enabled_hints)
            })

            logger.info(f"{fname}: Best Hints={enabled_hints}, PredCost={predicted_cost:.4f}, Latency={latency:.4f}s")

        except Exception as e:
            logger.error(f"Error executing {fname}: {e}")

    if not results:
        logger.warning("No results to save.")
        print("No results to display.")
        return

    df_results = pd.DataFrame(results)
    columns_order = ["filename", "best_idx", "best_combo", "predicted_cost", "latency", "hints"]
    df_results = df_results[columns_order]
    try:
        df_results.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"✅ Results saved to {OUTPUT_CSV}")
        print("\nResults Summary:")
        print(df_results.describe())
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
