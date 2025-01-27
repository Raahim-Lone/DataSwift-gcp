# optimizer.py

import asyncio
import asyncpg
import sqlglot
from sqlglot import exp
import logging
import numpy as np
import joblib
import os

# ======================================
# 📊 CONFIGURATION
# ======================================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Database Configuration
DB_CONFIG = {
    "user": "postgres",
    "password": "raahimlhere",
    "host": "localhost",
    "port": "6543",
    "database": "IMDB"
}

# Paths to Model Files
MODEL_DIR = '/Users/raahimlone/New_Data'
U_PATH = os.path.join(MODEL_DIR, 'U_best.npy')
V_PATH = os.path.join(MODEL_DIR, 'V_best.npy')
X_SCALED_PATH = os.path.join(MODEL_DIR, 'X_scaled.npz')
Y_SCALED_PATH = os.path.join(MODEL_DIR, 'Y_scaled.npz')
RESIDUAL_MODEL_PATH = os.path.join(MODEL_DIR, 'residual_model_xgb.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'residual_model_scaler.pkl')
POLY_PATH = os.path.join(MODEL_DIR, 'residual_model_poly.pkl')
HINTS_FILE_PATH = os.path.join(MODEL_DIR, 'hints.txt')

# ======================================
# 🛠️ GLOBALS (Loaded Once)
# ======================================
# We’ll load these when `init_optimizer()` is called.
U_best = None
V_best = None
B = None
residual_model = None
scaler = None
poly = None
HINTS = None

db_pool = None  # We'll store an asyncpg pool here.

# ======================================
# 🛠️ HELPER FUNCTIONS
# ======================================

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
        exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ,
        exp.Like, exp.ILike, exp.In, exp.Between
    )
    predicates = list(where_clause.find_all(predicate_types))
    return len(predicates)

def count_joins(where_clause, alias_mapping):
    if where_clause is None:
        return 0
    join_pairs = set()
    for predicate in where_clause.find_all(exp.EQ):
        left, right = predicate.left, predicate.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = alias_mapping.get(left.table)
            right_table = alias_mapping.get(right.table)
            if left_table and right_table and left_table != right_table:
                join_pairs.add(tuple(sorted([left_table, right_table])))
    return len(join_pairs)

def count_columns(stmt):
    columns = set()
    for column in stmt.find_all(exp.Column):
        columns.add(column.name)
    return len(columns)

def parse_sql(query_str):
    """
    Parse SQL query into 60 features.
    """
    try:
        stmt = sqlglot.parse_one(query_str, dialect='postgres')
        alias_mapping = build_alias_mapping(stmt)
        where_clause = stmt.find(exp.Where)

        features = {
            "num_predicates": count_predicates(where_clause),
            "num_joins": count_joins(where_clause, alias_mapping),
            "num_columns": count_columns(stmt),
            "num_tokens": len(query_str.split()),
            "num_chars": len(query_str),
            "has_group_by": 1.0 if stmt.args.get("group") else 0.0,
            "has_order_by": 1.0 if stmt.args.get("order") else 0.0,
        }

        # Add additional features if necessary to reach 60
        while len(features) < 60:
            features[f"extra_feature_{len(features)+1}"] = 0

        # Ensure exactly 60 features
        assert len(features) == 60, f"Expected 60 features, got {len(features)}"

        # Put them in a fixed order
        ordered_features = [
            features["num_predicates"],
            features["num_joins"],
            features["num_columns"],
            features["num_tokens"],
            features["num_chars"],
            features["has_group_by"],
            features["has_order_by"],
        ]
        for i in range(1, 60 - 7 + 1):
            ordered_features.append(features[f"extra_feature_{i}"])

        return ordered_features

    except Exception as e:
        logger.error(f"Failed to parse SQL query: {query_str[:50]}... | Error: {e}")
        return [0] * 60

def optimize_query(sql_query):
    """
    Given a SQL string, returns the same SQL string with an embedded hint
    that is predicted to yield the best performance.
    """
    try:
        # Parse
        features = parse_sql(sql_query)  # (60,)  
        A_novel = np.array([features])   # shape (1, 60)

        # 1) Matrix completion model to get base latency predictions
        W_hat_base = A_novel @ U_best      # shape: (1, rank)
        W_hat_base = W_hat_base @ V_best.T # shape: (1, d2)
        W_hat_base = W_hat_base @ B.T      # shape: (1, n_hints)
        W_hat_base = W_hat_base.flatten()  # shape: (n_hints,)

        # 2) Residual model
        num_hints = B.shape[0]
        query_features = np.repeat(A_novel, num_hints, axis=0)  # (n_hints, 60)
        hint_features = B.copy()                                # (n_hints, d2)

        combined_features = np.hstack([query_features, hint_features])
        combined_features_poly = poly.transform(combined_features)
        combined_features_scaled = scaler.transform(combined_features_poly)

        residuals_pred = residual_model.predict(combined_features_scaled)  # (n_hints,)

        # 3) Combine
        W_hat_adjusted = W_hat_base + residuals_pred

        # 4) Pick the best hint
        optimal_hint_idx = np.argmin(W_hat_adjusted)
        # Safety check
        if 0 <= optimal_hint_idx < len(HINTS):
            optimal_hint = HINTS[optimal_hint_idx]
            optimized_query = f"{sql_query} /*+ {optimal_hint} */"
            return optimized_query
        else:
            logger.error(f"Optimal hint index {optimal_hint_idx} out of range.")
            return sql_query  # fallback
    except Exception as e:
        logger.error(f"Error in optimize_query: {e}")
        # fallback: return the original query
        return sql_query

async def execute_optimized_query(sql_query):
    """
    Optimize the query, then execute it using our global db_pool.
    Returns a list of records (each record is a dict).
    """
    optimized = optimize_query(sql_query)
    logger.debug(f"Optimized query: {optimized}")

    # Execute
    async with db_pool.acquire() as conn:
        records = await conn.fetch(optimized)
        # Convert to list of dicts
        results = [dict(r) for r in records]
        return results

# ======================================
# 🚀 INITIALIZATION
# ======================================

async def init_optimizer():
    """
    Called once at startup to:
      - Connect to DB
      - Load model artifacts (U, V, B, residual_model, scaler, poly)
      - Load hints
      - Create an asyncpg pool
    """
    global U_best, V_best, B
    global residual_model, scaler, poly
    global HINTS
    global db_pool

    # 1) Connect to DB
    logger.info("🔗 Creating asyncpg pool...")
    db_pool = await asyncpg.create_pool(**DB_CONFIG)
    logger.info("✅ asyncpg connection pool created.")

    # 2) Load model artifacts
    try:
        U_best = np.load(U_PATH)
        logger.info("✅ Loaded U_best.npy")
    except Exception as e:
        logger.error(f"❌ Failed to load U_best.npy: {e}")
        raise

    try:
        V_best = np.load(V_PATH)
        logger.info("✅ Loaded V_best.npy")
    except Exception as e:
        logger.error(f"❌ Failed to load V_best.npy: {e}")
        raise

    try:
        with np.load(X_SCALED_PATH) as data:
            A = data['data']  # Not necessarily used directly, but confirm shape if needed
        logger.info("✅ Loaded X_scaled.npz")
    except Exception as e:
        logger.error(f"❌ Failed to load X_scaled.npz: {e}")
        raise

    try:
        with np.load(Y_SCALED_PATH) as data:
            B = data['Y']
        logger.info("✅ Loaded Y_scaled.npz")
    except Exception as e:
        logger.error(f"❌ Failed to load Y_scaled.npz: {e}")
        raise

    try:
        residual_model = joblib.load(RESIDUAL_MODEL_PATH)
        logger.info("✅ Loaded residual_model_xgb.pkl")
    except Exception as e:
        logger.error(f"❌ Failed to load residual_model_xgb.pkl: {e}")
        raise

    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Loaded residual_model_scaler.pkl")
    except Exception as e:
        logger.error(f"❌ Failed to load residual_model_scaler.pkl: {e}")
        raise

    try:
        poly = joblib.load(POLY_PATH)
        logger.info("✅ Loaded residual_model_poly.pkl")
    except Exception as e:
        logger.error(f"❌ Failed to load residual_model_poly.pkl: {e}")
        raise

    # 3) Load HINTS
    try:
        with open(HINTS_FILE_PATH, 'r') as f:
            HINTS = [line.strip() for line in f if line.strip()]
        logger.info(f"✅ Loaded {len(HINTS)} hints")
    except FileNotFoundError:
        logger.error("❌ Hints file not found. Using default hints.")
        HINTS = ["NO_HINT"]

    # Adjust the HINTS length if needed
    expected_num_hints = B.shape[0]
    if len(HINTS) < expected_num_hints:
        HINTS += ["NO_HINT"] * (expected_num_hints - len(HINTS))
    elif len(HINTS) > expected_num_hints:
        HINTS = HINTS[:expected_num_hints]
    elif len(HINTS) == expected_num_hints - 1:
        HINTS.append("NO_HINT")
    else:
        # If exactly matches or we have appended, it's fine
        pass

    logger.info(f"✅ Initialization of optimizer complete. Hints: {len(HINTS)}")

