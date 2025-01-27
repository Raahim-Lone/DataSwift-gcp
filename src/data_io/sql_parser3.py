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
    predicates = list(where_clause.find_all(*predicate_types))

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
    for predicate in where_clause.find_all(*join_predicate_types):
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

def count_aggregate_functions(select_expressions):
    """
    Count the number of aggregate functions in the SELECT clause.
    """
    aggregate_functions = {'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'MEDIAN', 'MODE'}
    count = 0
    for expr in select_expressions:
        if isinstance(expr, exp.Alias):
            expr = expr.this
        if isinstance(expr, exp.Func) and expr.name.upper() in aggregate_functions:
            count += 1
    return count

def count_logical_operators(where_clause):
    """
    Count the number of logical operators (AND, OR, NOT) in the WHERE clause.
    """
    if where_clause is None:
        return 0

    logical_ops = (exp.And, exp.Or, exp.Not)
    count = sum(1 for _ in where_clause.find_all(*logical_ops))
    return count

def count_comparison_operators(where_clause):
    """
    Count the number of comparison operators (=, >, <, >=, <=, !=) in the WHERE clause.
    """
    if where_clause is None:
        return 0

    comparison_ops = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ)
    count = sum(1 for _ in where_clause.find_all(*comparison_ops))
    return count

def count_group_by_columns(stmt):
    """
    Count the number of columns in the GROUP BY clause.
    """
    group = stmt.find(exp.Group)
    if group:
        return len(list(group.find_all(exp.Column)))
    return 0

def count_order_by_columns(stmt):
    """
    Count the number of columns in the ORDER BY clause.
    """
    order = stmt.find(exp.Order)
    if order:
        return len(list(order.find_all(exp.Ordered)))
    return 0

def count_nested_subqueries(stmt):
    """
    Count the number of nested subqueries within the main query.
    """
    # Exclude the main select statement
    subqueries = list(stmt.find_all(exp.Select))
    return max(0, len(subqueries) - 1)

def count_correlated_subqueries(stmt):
    """
    Check if there are any correlated subqueries.
    A correlated subquery references columns from the outer query.
    """
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
    """
    Count the number of CASE statements in the query.
    """
    return len(list(stmt.find_all(exp.Case)))

def count_union_operations(stmt):
    """
    Count the number of UNION or UNION ALL operations in the query.
    """
    return len(list(stmt.find_all(exp.Union)))

def parse_sql(query_str):
    """
    Parse SQL using sqlglot with Postgres dialect and extract enhanced features:
    - num_tables
    - num_joins
    - num_predicates
    - has_group_by
    - has_order_by
    - has_subquery
    - query_length
    - num_tokens
    - num_select_expressions
    - has_distinct
    - num_aggregate_functions
    - num_logical_operators
    - num_comparison_operators
    - num_group_by_columns
    - num_order_by_columns
    - num_nested_subqueries
    - has_correlated_subqueries
    - num_inner_joins
    - num_left_joins
    - num_right_joins
    - num_full_outer_joins
    - has_limit
    - has_union
    - num_union_operations
    - num_case_statements
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
        num_nested_subqueries = count_nested_subqueries(stmt)
        has_subquery = 1.0 if num_nested_subqueries > 0 else 0.0
        logger.debug(f"Has subquery: {has_subquery}")

        # Query Length Features
        query_length = len(query_str)
        num_tokens = len(list(stmt.walk()))
        logger.debug(f"Query length (chars): {query_length}")
        logger.debug(f"Number of tokens: {num_tokens}")

        # SELECT Clause Features
        select = stmt.find(exp.Select)
        select_expressions = select.expressions if select else []
        num_select_expressions = len(select_expressions)
        has_distinct = 1.0 if select and select.args.get("distinct") else 0.0
        num_aggregate_functions = count_aggregate_functions(select_expressions)
        logger.debug(f"Number of SELECT expressions: {num_select_expressions}")
        logger.debug(f"Has DISTINCT: {has_distinct}")
        logger.debug(f"Number of aggregate functions: {num_aggregate_functions}")

        # WHERE Clause Detailed Features
        num_logical_operators = count_logical_operators(where_clause)
        num_comparison_operators = count_comparison_operators(where_clause)
        logger.debug(f"Number of logical operators: {num_logical_operators}")
        logger.debug(f"Number of comparison operators: {num_comparison_operators}")

        # GROUP BY and ORDER BY Columns
        num_group_by_columns = count_group_by_columns(stmt)
        num_order_by_columns = count_order_by_columns(stmt)
        logger.debug(f"Number of GROUP BY columns: {num_group_by_columns}")
        logger.debug(f"Number of ORDER BY columns: {num_order_by_columns}")

        # Subquery Features
        has_correlated_subqueries = count_correlated_subqueries(stmt)
        logger.debug(f"Has correlated subqueries: {has_correlated_subqueries}")

        # JOIN Type Features (Explicit Joins)
        list_of_joins = list(stmt.find_all(exp.Join))
        num_inner_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'INNER'])
        num_left_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'LEFT'])
        num_right_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() == 'RIGHT'])
        num_full_outer_joins = len([j for j in list_of_joins if j.args.get('kind') and j.args.get('kind').upper() in ['FULL OUTER', 'FULLOUTER', 'FULL_OUTER']])
        logger.debug(f"Number of INNER JOINs: {num_inner_joins}")
        logger.debug(f"Number of LEFT JOINs: {num_left_joins}")
        logger.debug(f"Number of RIGHT JOINs: {num_right_joins}")
        logger.debug(f"Number of FULL OUTER JOINs: {num_full_outer_joins}")

        # Miscellaneous Features
        has_limit = 1.0 if stmt.find(exp.Limit) else 0.0
        has_union = 1.0 if stmt.find(exp.Union) else 0.0
        num_union_operations = count_union_operations(stmt)
        num_case_statements = count_case_statements(stmt)
        logger.debug(f"Has LIMIT: {has_limit}")
        logger.debug(f"Has UNION: {has_union}")
        logger.debug(f"Number of UNION operations: {num_union_operations}")
        logger.debug(f"Number of CASE statements: {num_case_statements}")

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

        logger.debug(f"Parsed SQL features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error parsing SQL: {e}")
        # Return a default feature set with zeros
        return {
            "num_tables": 0,
            "num_joins": 0,
            "num_predicates": 0,
            "has_group_by": 0.0,
            "has_order_by": 0.0,
            "has_subquery": 0.0,
            "query_length": 0,
            "num_tokens": 0,
            "num_select_expressions": 0,
            "has_distinct": 0.0,
            "num_aggregate_functions": 0,
            "num_logical_operators": 0,
            "num_comparison_operators": 0,
            "num_group_by_columns": 0,
            "num_order_by_columns": 0,
            "num_nested_subqueries": 0,
            "has_correlated_subqueries": 0.0,
            "num_inner_joins": 0,
            "num_left_joins": 0,
            "num_right_joins": 0,
            "num_full_outer_joins": 0,
            "has_limit": 0.0,
            "has_union": 0.0,
            "num_union_operations": 0,
            "num_case_statements": 0
        }
