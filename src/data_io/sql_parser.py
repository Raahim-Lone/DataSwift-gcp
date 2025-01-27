import logging
import re
import sqlglot
from sqlglot import exp
from sqlglot.tokens import Tokenizer

logger = logging.getLogger(__name__)

# ===============================================================
# 1) Feature Schema (EXACTLY 60) – Removed the 3 unused keys
# ===============================================================
FEATURE_SCHEMA_60 = [
    "num_tables",
    "total_joins",
    "num_inner_joins",
    "num_left_joins",
    "num_right_joins",
    "num_full_joins",
    "num_cross_joins",
    "num_semi_joins",
    "num_anti_joins",
    "num_other_joins",

    "num_union",
    "num_intersect",
    "num_except",

    "num_agg_count",
    "num_agg_sum",
    "num_agg_avg",
    "num_agg_min",
    "num_agg_max",
    "num_agg_other",

    "num_distinct",
    "num_limit",
    "num_offset",

    # function usage
    "func_count",
    "func_sum",
    "func_avg",
    "func_min",
    "func_max",
    "func_other",

    "num_columns_accessed",

    "num_predicates",
    "num_join_predicates",

    # keyword frequencies (27 total)
    "kw_select",
    "kw_from",
    "kw_where",
    "kw_join",
    "kw_inner",
    "kw_left",
    "kw_right",
    "kw_full",
    "kw_on",
    "kw_group_by",
    "kw_order_by",
    "kw_having",
    "kw_limit",
    "kw_offset",
    "kw_union",
    "kw_intersect",
    "kw_except",
    "kw_distinct",
    "kw_case",
    "kw_when",
    "kw_then",
    "kw_else",
    "kw_end",
    "kw_like",
    "kw_in",
    "kw_between",

    # query length
    "num_tokens",
    "num_chars",

    # Subquery flag
    "has_subquery"
]
assert len(FEATURE_SCHEMA_60) == 60, "Must have exactly 60 feature names."

# ===============================================================
# 2) Helper Functions
# ===============================================================

def build_alias_mapping(stmt):
    alias_mapping = {}
    for table in stmt.find_all(exp.Table):
        alias = table.alias_or_name
        name = table.name
        alias_mapping[alias] = name
    return alias_mapping

def count_join_types(stmt):
    join_types = {
        "INNER": 0,
        "LEFT": 0,
        "RIGHT": 0,
        "FULL": 0,
        "CROSS": 0,
        "LEFT_SEMI": 0,
        "LEFT_ANTI": 0,
        "OTHER": 0
    }
    for join in stmt.find_all(exp.Join):
        join_kind = join.args.get("kind", "INNER")
        join_type = join_kind.upper() if isinstance(join_kind, str) else "INNER"

        if "INNER" in join_type:
            join_types["INNER"] += 1
        elif "LEFT" in join_type and "SEM" in join_type:
            join_types["LEFT_SEMI"] += 1
        elif "LEFT" in join_type and "ANTI" in join_type:
            join_types["LEFT_ANTI"] += 1
        elif "LEFT" in join_type:
            join_types["LEFT"] += 1
        elif "RIGHT" in join_type:
            join_types["RIGHT"] += 1
        elif "FULL" in join_type:
            join_types["FULL"] += 1
        elif "CROSS" in join_type:
            join_types["CROSS"] += 1
        else:
            join_types["OTHER"] += 1
    return join_types

def count_predicates(where_clause):
    if where_clause is None:
        return 0
    predicate_types = (
        exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE,
        exp.NEQ, exp.Like, exp.ILike, exp.In, exp.Between
    )
    return len(list(where_clause.find_all(predicate_types)))

def count_joins(where_clause, alias_mapping):
    if where_clause is None:
        return 0
    join_pairs = set()
    for eq_expr in where_clause.find_all(exp.EQ):
        left = eq_expr.left
        right = eq_expr.right
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = alias_mapping.get(left.table)
            right_table = alias_mapping.get(right.table)
            if left_table and right_table and left_table != right_table:
                pair = tuple(sorted([left_table, right_table]))
                join_pairs.add(pair)
    return len(join_pairs)

def count_set_operations(stmt):
    set_ops = {"UNION": 0, "INTERSECT": 0, "EXCEPT": 0}
    for op in stmt.find_all(exp.SetOperation):
        op_type = op.__class__.__name__.upper()
        if "UNION" in op_type:
            set_ops["UNION"] += 1
        elif "INTERSECT" in op_type:
            set_ops["INTERSECT"] += 1
        elif "EXCEPT" in op_type:
            set_ops["EXCEPT"] += 1
    return set_ops

def count_aggregation_functions(stmt):
    aggregation_types = {
        "COUNT": exp.Count,
        "SUM": exp.Sum,
        "AVG": exp.Avg,
        "MIN": exp.Min,
        "MAX": exp.Max
    }
    agg_counts = {k: 0 for k in aggregation_types.keys()}
    agg_counts["OTHER"] = 0
    for func in stmt.find_all(exp.Func):
        fname = (func.name or "").upper()
        if fname in aggregation_types:
            agg_counts[fname] += 1
        else:
            agg_counts["OTHER"] += 1
    return agg_counts

def count_distinct(stmt):
    dist_count = 0
    for sel in stmt.find_all(exp.Select):
        if sel.args.get("distinct"):
            dist_count += 1
    return dist_count

def count_limit_offset(stmt):
    limit = sum(1 for _ in stmt.find_all(exp.Limit))
    offset = sum(1 for _ in stmt.find_all(exp.Offset))
    return {"limit": limit, "offset": offset}

def count_functions(stmt):
    func_usage = {}
    for func in stmt.find_all(exp.Func):
        fname = (func.name or "unknown").lower()
        func_usage[fname] = func_usage.get(fname, 0) + 1
    return func_usage

def count_columns(stmt):
    cols = set()
    for c in stmt.find_all(exp.Column):
        cols.add(c.name)
    return len(cols)

def count_keywords(query_str):
    import re
    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT",
        "FULL", "ON", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET",
        "UNION", "INTERSECT", "EXCEPT", "DISTINCT", "CASE", "WHEN", "THEN",
        "ELSE", "END", "LIKE", "IN", "BETWEEN"
    ]
    kw_counts = {kw.lower().replace(" ", "_"): 0 for kw in keywords}
    for kw in keywords:
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        matches = pattern.findall(query_str)
        kw_counts[kw.lower().replace(" ", "_")] = len(matches)
    return kw_counts

def count_query_length(query_str):
    if not query_str.strip():
        return {"num_tokens": 0, "num_chars": 0}
    try:
        tokenizer = Tokenizer(dialect='postgres')
        tokens = tokenizer.tokenize(query_str)
        return {"num_tokens": len(tokens), "num_chars": len(query_str)}
    except Exception:
        return {"num_tokens": 0, "num_chars": len(query_str)}

def count_subquery_depth(stmt, current_depth=0, max_depth=0):
    for s in stmt.find_all(exp.Select):
        if s is not stmt:
            depth, max_depth = count_subquery_depth(s, current_depth+1, max_depth)
            if depth > max_depth:
                max_depth = depth
    return current_depth, max_depth

# ===============================================================
# 3) Main Function to Return EXACT 60 Features
# ===============================================================
def parse_sql_60_features(query_str):
    """
    Parse a query -> EXACT 60 features.
    We removed 'max_subquery_depth', 'has_group_by', 'has_order_by'.
    """
    feats_60 = {k: 0 for k in FEATURE_SCHEMA_60}
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        stmt = statements[0]

        # Build alias map
        alias_map = build_alias_mapping(stmt)

        # 1) Table + Join info
        tables = [node.alias_or_name for node in stmt.find_all(exp.Table)]
        feats_60["num_tables"] = len(tables)

        jt = count_join_types(stmt)
        feats_60["total_joins"]      = sum(jt.values())
        feats_60["num_inner_joins"]  = jt["INNER"]
        feats_60["num_left_joins"]   = jt["LEFT"]
        feats_60["num_right_joins"]  = jt["RIGHT"]
        feats_60["num_full_joins"]   = jt["FULL"]
        feats_60["num_cross_joins"]  = jt["CROSS"]
        feats_60["num_semi_joins"]   = jt["LEFT_SEMI"]
        feats_60["num_anti_joins"]   = jt["LEFT_ANTI"]
        feats_60["num_other_joins"]  = jt["OTHER"]

        # 2) Set Operations
        set_ops = count_set_operations(stmt)
        feats_60["num_union"]     = set_ops["UNION"]
        feats_60["num_intersect"] = set_ops["INTERSECT"]
        feats_60["num_except"]    = set_ops["EXCEPT"]

        # 3) Aggregations
        agg_counts = count_aggregation_functions(stmt)
        feats_60["num_agg_count"] = agg_counts["COUNT"]
        feats_60["num_agg_sum"]   = agg_counts["SUM"]
        feats_60["num_agg_avg"]   = agg_counts["AVG"]
        feats_60["num_agg_min"]   = agg_counts["MIN"]
        feats_60["num_agg_max"]   = agg_counts["MAX"]
        feats_60["num_agg_other"] = agg_counts["OTHER"]

        # 4) DISTINCT, LIMIT, OFFSET
        feats_60["num_distinct"] = count_distinct(stmt)
        lo = count_limit_offset(stmt)
        feats_60["num_limit"]  = lo["limit"]
        feats_60["num_offset"] = lo["offset"]

        # 5) Functions -> 6 keys
        func_usage = count_functions(stmt)
        other_funcs = 0
        for fname, cval in func_usage.items():
            if fname not in ["count", "sum", "avg", "min", "max"]:
                other_funcs += cval
        feats_60["func_count"] = func_usage.get("count", 0)
        feats_60["func_sum"]   = func_usage.get("sum", 0)
        feats_60["func_avg"]   = func_usage.get("avg", 0)
        feats_60["func_min"]   = func_usage.get("min", 0)
        feats_60["func_max"]   = func_usage.get("max", 0)
        feats_60["func_other"] = other_funcs

        # 6) Columns + Predicates
        feats_60["num_columns_accessed"] = count_columns(stmt)

        where_clause = stmt.find(exp.Where)
        feats_60["num_predicates"]      = count_predicates(where_clause)
        feats_60["num_join_predicates"] = count_joins(where_clause, alias_map)

        # 7) Keyword Frequencies
        kw = count_keywords(query_str)
        feats_60["kw_select"]    = kw["select"]
        feats_60["kw_from"]      = kw["from"]
        feats_60["kw_where"]     = kw["where"]
        feats_60["kw_join"]      = kw["join"]
        feats_60["kw_inner"]     = kw["inner"]
        feats_60["kw_left"]      = kw["left"]
        feats_60["kw_right"]     = kw["right"]
        feats_60["kw_full"]      = kw["full"]
        feats_60["kw_on"]        = kw["on"]
        feats_60["kw_group_by"]  = kw["group_by"]
        feats_60["kw_order_by"]  = kw["order_by"]
        feats_60["kw_having"]    = kw["having"]
        feats_60["kw_limit"]     = kw["limit"]
        feats_60["kw_offset"]    = kw["offset"]
        feats_60["kw_union"]     = kw["union"]
        feats_60["kw_intersect"] = kw["intersect"]
        feats_60["kw_except"]    = kw["except"]
        feats_60["kw_distinct"]  = kw["distinct"]
        feats_60["kw_case"]      = kw["case"]
        feats_60["kw_when"]      = kw["when"]
        feats_60["kw_then"]      = kw["then"]
        feats_60["kw_else"]      = kw["else"]
        feats_60["kw_end"]       = kw["end"]
        feats_60["kw_like"]      = kw["like"]
        feats_60["kw_in"]        = kw["in"]
        feats_60["kw_between"]   = kw["between"]

        # 8) Query length
        qlen = count_query_length(query_str)
        feats_60["num_tokens"] = qlen["num_tokens"]
        feats_60["num_chars"]  = qlen["num_chars"]

        # 9) Subquery (just has_subquery)
        _, max_sub_d = count_subquery_depth(stmt)
        feats_60["has_subquery"] = 1.0 if max_sub_d > 0 else 0.0

    except Exception as ex:
        logger.error(f"Error parsing SQL: {ex}\nQuery: {query_str[:100]}")
        # If something fails, we return the dict of zeros

    return feats_60


'''
import logging
import sqlglot
from sqlglot import exp
from sqlglot.tokens import Tokenizer  # Correctly import the Tokenizer class

logger = logging.getLogger(__name__)

def build_alias_mapping(stmt):
    """
    Build a mapping from table aliases to table names.
    """
    alias_mapping = {}
    for table in stmt.find_all(exp.Table):
        alias = table.alias_or_name
        name = table.name
        alias_mapping[alias] = name
        logger.debug(f"Mapping alias '{alias}' to table '{name}'")
    return alias_mapping

def count_predicates(where_clause):
    """
    Count the number of predicate expressions in the WHERE clause.
    Predicate types include EQ, GT, LT, GTE, LTE, NEQ, Like, ILike, In, Between.
    """
    if where_clause is None:
        return 0

    # Define the predicate expression types to count
    predicate_types = (
        exp.EQ,
        exp.GT,
        exp.LT,
        exp.GTE,
        exp.LTE,
        exp.NEQ,
        exp.Like,
        exp.ILike,
        exp.In,
        exp.Between,
    )

    # Use find_all to locate all instances of these predicate types
    predicates = list(where_clause.find_all(predicate_types))

    return len(predicates)

def count_joins(where_clause, alias_mapping):
    """
    Count the number of unique join conditions in the WHERE clause.
    A join condition is defined as an equality predicate between two different tables.
    """
    if where_clause is None:
        return 0

    join_pairs = set()

    # Define only EQ predicates for joins
    join_predicate_types = (exp.EQ,)

    # Iterate over all EQ predicates in the WHERE clause
    for predicate in where_clause.find_all(join_predicate_types):
        left = predicate.left
        right = predicate.right

        # Check if both sides are columns
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            # Extract table aliases or names
            left_table_alias = left.table
            right_table_alias = right.table

            # Get actual table names from alias mapping
            left_table_name = alias_mapping.get(left_table_alias)
            right_table_name = alias_mapping.get(right_table_alias)

            # Log the predicate being processed
            logger.debug(f"Processing predicate: {left.sql()} = {right.sql()} (Tables: {left_table_name}, {right_table_name})")

            # Ensure both tables are present and different
            if left_table_name and right_table_name and left_table_name != right_table_name:
                # Create a sorted tuple to avoid counting (A, B) and (B, A) separately
                pair = tuple(sorted([left_table_name, right_table_name]))
                if pair not in join_pairs:
                    join_pairs.add(pair)
                    logger.debug(f"Identified unique join pair: {pair}")
                else:
                    logger.debug(f"Duplicate join pair detected and ignored: {pair}")
            else:
                logger.debug(f"Ignored predicate as it does not connect two different tables: {left.sql()} = {right.sql()}")

    logger.debug(f"Total unique joins counted: {len(join_pairs)}")
    return len(join_pairs)

def count_join_types(stmt):
    """
    Count the number of each type of JOIN in the query.
    """
    join_types = {
        "INNER": 0,
        "LEFT": 0,
        "RIGHT": 0,
        "FULL": 0,
        "CROSS": 0,
        "LEFT_SEMI": 0,
        "LEFT_ANTI": 0,
        "OTHER": 0
    }

    for join in stmt.find_all(exp.Join):
        join_kind = join.args.get("kind", "INNER")
        join_type = join_kind.upper() if isinstance(join_kind, str) else "INNER"

        # Map specific join types to predefined categories
        if "INNER" in join_type:
            join_types["INNER"] += 1
        elif "LEFT" in join_type and "SEM" in join_type:
            join_types["LEFT_SEMI"] += 1
        elif "LEFT" in join_type and "ANTI" in join_type:
            join_types["LEFT_ANTI"] += 1
        elif "LEFT" in join_type:
            join_types["LEFT"] += 1
        elif "RIGHT" in join_type:
            join_types["RIGHT"] += 1
        elif "FULL" in join_type:
            join_types["FULL"] += 1
        elif "CROSS" in join_type:
            join_types["CROSS"] += 1
        else:
            join_types["OTHER"] += 1
        logger.debug(f"Found JOIN type: {join_type}")

    return join_types

def count_set_operations(stmt):
    """
    Count the number of set operations: UNION, INTERSECT, EXCEPT.
    """
    set_operations = {
        "UNION": 0,
        "INTERSECT": 0,
        "EXCEPT": 0
    }

    for operation in stmt.find_all(exp.SetOperation):
        op_type = operation.__class__.__name__.upper()
        if "UNION" in op_type:
            set_operations["UNION"] += 1
        elif "INTERSECT" in op_type:
            set_operations["INTERSECT"] += 1
        elif "EXCEPT" in op_type:
            set_operations["EXCEPT"] += 1
        logger.debug(f"Found set operation: {op_type}")

    return set_operations

def count_aggregation_functions(stmt):
    """
    Count the number of aggregation functions like COUNT, SUM, AVG, MIN, MAX.
    """
    aggregation_types = {
        "COUNT": exp.Count,
        "SUM": exp.Sum,
        "AVG": exp.Avg,
        "MIN": exp.Min,
        "MAX": exp.Max
    }

    aggregation_counts = {key: 0 for key in aggregation_types.keys()}
    aggregation_counts['OTHER'] = 0

    for func in stmt.find_all(exp.Func):
        # Extract function name, handle cases where name might be None
        func_name = func.name.upper() if func.name else func.__class__.__name__.upper()

        if func_name in aggregation_types:
            aggregation_counts[func_name] += 1
            logger.debug(f"Found aggregation function: {func_name}")
        else:
            aggregation_counts['OTHER'] += 1
            logger.debug(f"Found non-standard aggregation function: {func_name}")

    return aggregation_counts

def count_distinct(stmt):
    """
    Check if DISTINCT is used and count its occurrences.
    """
    distinct_count = 0
    for select in stmt.find_all(exp.Select):
        if select.args.get("distinct"):
            distinct_count += 1
            logger.debug("Found DISTINCT in SELECT clause.")
    return distinct_count

def count_limit_offset(stmt):
    """
    Count the number of LIMIT and OFFSET clauses.
    """
    limit = 0
    offset = 0
    for expr in stmt.find_all(exp.Limit):
        limit += 1
        logger.debug("Found LIMIT clause.")
    for expr in stmt.find_all(exp.Offset):
        offset += 1
        logger.debug("Found OFFSET clause.")
    return {"limit": limit, "offset": offset}

def count_functions(stmt):
    """
    Count the number of SQL functions used, categorized by type.
    """
    function_counts = {}
    for func in stmt.find_all(exp.Func):
        func_name = func.name.lower() if func.name else func.__class__.__name__.lower()
        function_counts[func_name] = function_counts.get(func_name, 0) + 1
        logger.debug(f"Found function: {func_name}")
    return function_counts

def count_columns(stmt):
    """
    Count the number of columns accessed in SELECT, WHERE, GROUP BY, ORDER BY clauses.
    """
    columns = set()
    for column in stmt.find_all(exp.Column):
        columns.add(column.name)
    logger.debug(f"Total unique columns accessed: {len(columns)}")
    return len(columns)

def count_keywords(query_str):
    """
    Count the frequency of specific SQL keywords in the query string.
    """
    import re
    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT",
        "FULL", "ON", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
        "OFFSET", "UNION", "INTERSECT", "EXCEPT", "DISTINCT", "CASE",
        "WHEN", "THEN", "ELSE", "END", "LIKE", "IN", "BETWEEN"
    ]
    keyword_counts = {kw.lower().replace(" ", "_"): 0 for kw in keywords}
    for kw in keywords:
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        matches = pattern.findall(query_str)
        kw_key = kw.lower().replace(" ", "_")
        keyword_counts[kw_key] = len(matches)
        if len(matches) > 0:
            logger.debug(f"Keyword '{kw}' occurs {len(matches)} times.")
    return keyword_counts

def count_query_length(query_str):
    """
    Calculate the length of the query in terms of tokens and characters.
    """
    if not query_str.strip():
        logger.debug("Empty query string.")
        return {"num_tokens": 0, "num_chars": 0}

    try:
        tokenizer = Tokenizer(dialect='postgres')  # Instantiate the Tokenizer
        tokens = tokenizer.tokenize(query_str)     # Tokenize the query
        num_tokens = len(tokens)
        num_chars = len(query_str)
        logger.debug(f"Query has {num_tokens} tokens and {num_chars} characters.")
        return {"num_tokens": num_tokens, "num_chars": num_chars}
    except Exception as e:
        logger.error(f"Error tokenizing SQL: {e}")
        return {"num_tokens": 0, "num_chars": len(query_str)}

def count_subquery_depth(stmt, current_depth=0, max_depth=0):
    """
    Recursively determine the maximum depth of nested subqueries.
    """
    for select in stmt.find_all(exp.Select):
        if select is not stmt:
            depth, max_depth = count_subquery_depth(select, current_depth + 1, max_depth)
            if depth > max_depth:
                max_depth = depth
    return current_depth, max_depth

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract a wide range of features.
    """
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")

        stmt = statements[0]

        # Build alias mapping
        alias_mapping = build_alias_mapping(stmt)

        # Extract structural features
        tables = [node.alias_or_name for node in stmt.find_all(exp.Table)]
        num_tables = len(tables)
        logger.debug(f"Number of tables: {num_tables}")

        join_types = count_join_types(stmt)
        total_joins = sum(join_types.values())
        logger.debug(f"JOIN types: {join_types}")

        set_ops = count_set_operations(stmt)
        logger.debug(f"Set operations: {set_ops}")

        agg_funcs = count_aggregation_functions(stmt)
        logger.debug(f"Aggregation functions: {agg_funcs}")

        num_distinct = count_distinct(stmt)
        logger.debug(f"Number of DISTINCT clauses: {num_distinct}")

        limit_offset = count_limit_offset(stmt)
        logger.debug(f"LIMIT clauses: {limit_offset['limit']}, OFFSET clauses: {limit_offset['offset']}")

        func_usage = count_functions(stmt)
        logger.debug(f"Function usage: {func_usage}")

        num_columns = count_columns(stmt)
        logger.debug(f"Number of columns accessed: {num_columns}")

        # Extract WHERE predicates
        where_clause = stmt.find(exp.Where)
        num_predicates = count_predicates(where_clause)
        logger.debug(f"Number of predicates: {num_predicates}")

        # Extract JOIN predicates (implicit joins)
        num_join_predicates = count_joins(where_clause, alias_mapping)
        logger.debug(f"Number of unique join predicates: {num_join_predicates}")

        # Extract keyword frequencies
        keyword_freq = count_keywords(query_str)
        logger.debug(f"Keyword frequencies: {keyword_freq}")

        # Extract query length
        query_length = count_query_length(query_str)
        logger.debug(f"Query length: {query_length}")

        # Extract subquery depth
        _, max_subquery_depth = count_subquery_depth(stmt)
        has_subquery = 1.0 if max_subquery_depth > 0 else 0.0
        logger.debug(f"Maximum subquery depth: {max_subquery_depth}")

        # Enhanced Flag Detection
        # Check if 'group' and 'order' are present in the AST
        has_group_by = 1.0 if stmt.args.get("group") else 0.0
        has_order_by = 1.0 if stmt.args.get("order") else 0.0
        logger.debug(f"Has GROUP BY: {has_group_by}")
        logger.debug(f"Has ORDER BY: {has_order_by}")

        # Alternatively, you can use find_all for robustness
        # has_group_by = 1.0 if any(stmt.find_all(exp.Group)) else 0.0
        # has_order_by = 1.0 if any(stmt.find_all(exp.Order)) else 0.0

        # Compile all features into a dictionary
        features = {
            # Structural Features
            "num_tables": num_tables,
            "total_joins": total_joins,
            "num_inner_joins": join_types.get("INNER", 0),
            "num_left_joins": join_types.get("LEFT", 0),
            "num_right_joins": join_types.get("RIGHT", 0),
            "num_full_joins": join_types.get("FULL", 0),
            "num_cross_joins": join_types.get("CROSS", 0),
            "num_semi_joins": join_types.get("LEFT_SEMI", 0),
            "num_anti_joins": join_types.get("LEFT_ANTI", 0),
            "num_other_joins": join_types.get("OTHER", 0),
            "num_union": set_ops.get("UNION", 0),
            "num_intersect": set_ops.get("INTERSECT", 0),
            "num_except": set_ops.get("EXCEPT", 0),

            # Aggregation Features
            "num_agg_count": agg_funcs.get("COUNT", 0),
            "num_agg_sum": agg_funcs.get("SUM", 0),
            "num_agg_avg": agg_funcs.get("AVG", 0),
            "num_agg_min": agg_funcs.get("MIN", 0),
            "num_agg_max": agg_funcs.get("MAX", 0),
            "num_agg_other": agg_funcs.get("OTHER", 0),

            # DISTINCT and LIMIT
            "num_distinct": num_distinct,
            "num_limit": limit_offset.get("limit", 0),
            "num_offset": limit_offset.get("offset", 0),

            # Function Usage
            **{f"func_{k}": v for k, v in func_usage.items()},

            # Columns Accessed
            "num_columns_accessed": num_columns,

            # Predicates and Joins
            "num_predicates": num_predicates,
            "num_join_predicates": num_join_predicates,

            # Keyword Frequencies
            **{f"kw_{k}": v for k, v in keyword_freq.items()},

            # Query Length
            "num_tokens": query_length["num_tokens"],
            "num_chars": query_length["num_chars"],

            # Subquery Features
            "has_subquery": has_subquery,
            "max_subquery_depth": max_subquery_depth,

            # Flags
            "has_group_by": has_group_by,
            "has_order_by": has_order_by
        }

        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}\nQuery: {query_str[:100]}...")
        return {}



'''
'''
import logging
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

def build_alias_mapping(stmt):
    """
    Build a mapping from table aliases to table names.
    """
    alias_mapping = {}
    for table in stmt.find_all(exp.Table):
        alias = table.alias_or_name
        name = table.name
        alias_mapping[alias] = name
        logger.debug(f"Mapping alias '{alias}' to table '{name}'")
    return alias_mapping

def count_predicates(where_clause):
    """
    Count the number of predicate expressions in the WHERE clause.
    Predicate types include EQ, GT, LT, GTE, LTE, NEQ, Like, ILike, In, Between.
    """
    if where_clause is None:
        return 0

    # Define the predicate expression types to count
    predicate_types = (
        exp.EQ,
        exp.GT,
        exp.LT,
        exp.GTE,
        exp.LTE,
        exp.NEQ,
        exp.Like,
        exp.ILike,
        exp.In,
        exp.Between,
    )

    # Use find_all to locate all instances of these predicate types
    predicates = list(where_clause.find_all(predicate_types))

    return len(predicates)

def count_joins(where_clause, alias_mapping):
    """
    Count the number of unique join conditions in the WHERE clause.
    A join condition is defined as an equality predicate between two different tables.
    """
    if where_clause is None:
        return 0

    join_pairs = set()

    # Define only EQ predicates for joins
    join_predicate_types = (exp.EQ,)

    # Iterate over all EQ predicates in the WHERE clause
    for predicate in where_clause.find_all(join_predicate_types):
        left = predicate.left
        right = predicate.right

        # Check if both sides are columns
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            # Extract table aliases or names
            left_table_alias = left.table
            right_table_alias = right.table

            # Get actual table names from alias mapping
            left_table_name = alias_mapping.get(left_table_alias)
            right_table_name = alias_mapping.get(right_table_alias)

            # Log the predicate being processed
            logger.debug(f"Processing predicate: {left.sql()} = {right.sql()} (Tables: {left_table_name}, {right_table_name})")

            # Ensure both tables are present and different
            if left_table_name and right_table_name and left_table_name != right_table_name:
                # Create a sorted tuple to avoid counting (A, B) and (B, A) separately
                pair = tuple(sorted([left_table_name, right_table_name]))
                if pair not in join_pairs:
                    join_pairs.add(pair)
                    logger.debug(f"Identified unique join pair: {pair}")
                else:
                    logger.debug(f"Duplicate join pair detected and ignored: {pair}")
            else:
                logger.debug(f"Ignored predicate as it does not connect two different tables: {left.sql()} = {right.sql()}")

    logger.debug(f"Total unique joins counted: {len(join_pairs)}")
    return len(join_pairs)

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    """
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")

        stmt = statements[0]

        # Build alias mapping
        alias_mapping = build_alias_mapping(stmt)

        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(exp.Table)]
        num_tables = len(tables)
        logger.debug(f"Number of tables: {num_tables}")

        # Extract WHERE predicates
        where_clause = stmt.find(exp.Where)
        num_predicates = count_predicates(where_clause)
        logger.debug(f"Number of predicates: {num_predicates}")

        # Extract JOIN predicates (implicit joins)
        num_joins = count_joins(where_clause, alias_mapping)
        logger.debug(f"Number of unique joins: {num_joins}")

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(exp.Group) else 0.0
        logger.debug(f"Has GROUP BY: {has_group_by}")

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(exp.Order) else 0.0
        logger.debug(f"Has ORDER BY: {has_order_by}")

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(exp.Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0
        logger.debug(f"Has subquery: {has_subquery}")

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''

'''
import logging
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

def count_predicates(where_clause):
    """
    Count the number of predicate expressions in the WHERE clause.
    Predicate types include EQ, GT, LT, GTE, LTE, NEQ, Like, ILike, In, Between.
    """
    if where_clause is None:
        return 0

    # Define the predicate expression types to count
    predicate_types = (
        exp.EQ,
        exp.GT,
        exp.LT,
        exp.GTE,
        exp.LTE,
        exp.NEQ,
        exp.Like,
        exp.ILike,
        exp.In,
        exp.Between,
    )

    # Use find_all to locate all instances of these predicate types
    predicates = list(where_clause.find_all(predicate_types))

    return len(predicates)

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    """
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")

        stmt = statements[0]

        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(exp.Table)]
        num_tables = len(tables)

        # Extract JOIN clauses
        joins = list(stmt.find_all(exp.Join))
        num_joins = len(joins)

        # Extract WHERE predicates
        where_clause = stmt.find(exp.Where)
        num_predicates = count_predicates(where_clause)

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(exp.Group) else 0.0

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(exp.Order) else 0.0

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(exp.Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''
'''
import logging
import sqlglot
from sqlglot.expressions import Table, Join, Where, Select, Group, Order, Condition, And, Or, Like, In, Column

logger = logging.getLogger(__name__)

def count_predicates(node):
    """
    Recursively count predicate expressions in WHERE clauses.
    """
    if node is None:
        return 0

    count = 0

    # Check if the node represents a predicate directly
    if node.key in {"eq", "gt", "lt", "gte", "lte", "neq", "like", "in", "between", "ilike"}:
        count += 1

    # Recursively process nested logical expressions (AND, OR)
    if isinstance(node, (And, Or)):
        for arg in node.args.values():
            count += count_predicates(arg)

    # Also, handle nested expressions directly
    if hasattr(node, "expressions"):
        for expr in node.expressions:
            count += count_predicates(expr)

    return count

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features.
    """
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        
        stmt = statements[0]

        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(Table)]
        num_tables = len(tables)

        # Extract JOIN clauses
        joins = list(stmt.find_all(Join))
        num_joins = len(joins)

        # Extract WHERE predicates
        where_clause = stmt.find(Where)
        if where_clause:
            print("WHERE Clause Detected:")
            print(where_clause.sql())  # Print raw WHERE SQL
            print("WHERE Clause Nodes:")
            for node in where_clause.find_all():
                print(f"{node} -> {type(node)}")

        num_predicates = count_predicates(where_clause)

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(Group) else 0.0

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(Order) else 0.0

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''
'''
import logging
import sqlglot
from sqlglot.expressions import Table, Join, Where, Select, Group, Order, Condition, And, Or, Like, In, Column

logger = logging.getLogger(__name__)

def count_predicates(node):
    """
    Recursively count predicate expressions in WHERE clauses.
    """
    if node is None:
        return 0
    
    count = 0
    if isinstance(node, (Condition, Like, In)):
        count += 1
    elif isinstance(node, (And, Or)):
        # Recursively traverse AND/OR nodes to find nested predicates
        count += sum(count_predicates(arg) for arg in node.args.values())
    return count

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    """
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        
        stmt = statements[0]
        
        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(Table)]
        num_tables = len(tables)

        # Extract JOIN clauses
        joins = list(stmt.find_all(Join))
        num_joins = len(joins)

        # Extract WHERE predicates
        where_clause = stmt.find(Where)
        num_predicates = count_predicates(where_clause)

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(Group) else 0.0

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(Order) else 0.0

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''
'''
import logging
import sqlglot
from sqlglot.expressions import Table, Join, Where, Select, Group, Order, Condition

logger = logging.getLogger(__name__)

def parse_sql(query_str):
    try:
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        
        stmt = statements[0]
        
        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(Table)]
        num_tables = len(tables)

        # Extract JOIN clauses
        joins = list(stmt.find_all(Join))
        num_joins = len(joins)

        # Extract WHERE predicates
        num_predicates = 0
        where_clause = stmt.find(Where)
        if where_clause:
            for node in where_clause.find_all():
                if isinstance(node, sqlglot.expressions.Condition):
                    num_predicates += 1

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(Group) else 0.0

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(Order) else 0.0

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''

'''
import logging
import sqlglot
from sqlglot.expressions import Table, Join, Where, Select, Group, Order

logger = logging.getLogger(__name__)

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    """
    try:
        # Use dialect='postgres' to handle Postgres-specific syntax
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        
        stmt = statements[0]  # Focus on the first parsed statement
        
        # Extract table names
        tables = [node.alias_or_name for node in stmt.find_all(Table)]
        num_tables = len(tables)

        # Extract JOIN clauses
        joins = list(stmt.find_all(Join))
        num_joins = len(joins)

        # Extract WHERE predicates
        num_predicates = 0
        where_clause = stmt.find(Where)
        if where_clause:
            num_predicates = sum(
                1 for node in where_clause.find_all()
                if node.key in {"eq", "gt", "lt", "gte", "lte", "neq", "like", "in", "between", "ilike"}
            )

        # Check for GROUP BY clause
        has_group_by = 1.0 if stmt.find(Group) else 0.0

        # Check for ORDER BY clause
        has_order_by = 1.0 if stmt.find(Order) else 0.0

        # Check for subqueries
        subqueries = sum(
            1 for node in stmt.find_all(Select) if node is not stmt
        )
        has_subquery = 1.0 if subqueries > 0 else 0.0

        # Log and return extracted features
        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''

'''
import logging
import sqlglot

logger = logging.getLogger(__name__)

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    """
    try:
        # Use dialect='postgres' to handle Postgres-specific syntax
        statements = sqlglot.parse(query_str, dialect='postgres')
        if not statements:
            raise ValueError("No statements parsed.")
        stmt = statements[0]

        tables = stmt.tables
        num_tables = len(tables)

        joins = [node for node in stmt.find_all("Join")]
        num_joins = len(joins)

        where_expr = stmt.args.get("where")
        num_predicates = 0
        if where_expr:
            for node in where_expr.find_all():
                if node.is_type("EQ", "GT", "LT", "GTE", "LTE", "NEQ", "LIKE", "IN", "BETWEEN", "ILIKE"):
                    num_predicates += 1

        has_group_by = 1.0 if stmt.args.get("group") else 0.0
        has_order_by = 1.0 if stmt.args.get("order") else 0.0

        # subqueries: SELECT nodes not the root
        subqueries = 0
        for node in stmt.find_all("Select"):
            if node is not stmt:
                subqueries += 1
        has_subquery = 1.0 if subqueries > 0 else 0.0

        features = {
            "num_tables": num_tables,
            "num_joins": num_joins,
            "num_predicates": num_predicates,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_subquery": has_subquery
        }
        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0
        }
'''