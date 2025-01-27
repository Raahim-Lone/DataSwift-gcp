
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
