import os
import psycopg2
import pandas as pd
import json
import re
import concurrent.futures
from psycopg2 import sql

# Configuration
DB_CONFIG = {
    'dbname': 'IMDB',
    'user': 'postgres',       
    'password': 'raahimlhere',   
    'host': 'localhost',  # later update for Cloud SQL
    'port': '6543',       
}

QUERIES_DIR = '/Users/raahimlone/imdb_pg_dataset-1'
HINTS_FILE = '/Users/raahimlone/project_root/hints.json'
OUTPUT_CSV = '/Users/raahimlone/rahhh/Data_Gathering/query_latency_testrah.csv'

def load_queries(directory):
    queries = {}
    for filename in os.listdir(directory):
        if filename.endswith('.sql'):
            with open(os.path.join(directory, filename), 'r') as file:
                queries[filename] = file.read()
    return queries

def load_hints(file_path):
    if not os.path.exists(file_path):
        print(f"Hints file '{file_path}' does not exist. Proceeding without hints.")
        return {}
    with open(file_path, 'r') as file:
        hints = json.load(file)
    return hints

def set_configurations(cursor, configurations):
    for param, value in configurations.items():
        set_command = sql.SQL("SET {} TO %s;").format(sql.Identifier(param))
        cursor.execute(set_command, [value])

def reset_configurations(cursor):
    cursor.execute("RESET ALL;")

def execute_explain_analyze(cursor, query):
    explain_query = f"EXPLAIN (ANALYZE, VERBOSE, BUFFERS) {query}"
    cursor.execute(explain_query)
    explain_output = cursor.fetchall()
    explain_text = "\n".join(row[0] for row in explain_output)
    latency = parse_explain_analyze(explain_text)
    return latency, explain_text

def parse_explain_analyze(explain_text):
    match = re.search(r"Execution Time:\s+([\d\.]+)\s+ms", explain_text)
    if match:
        return float(match.group(1)) / 1000  # convert ms to sec
    return None

def run_query(q_name, query, hint_name, configurations):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()

        if configurations:
            set_configurations(cursor, configurations)
        else:
            reset_configurations(cursor)

        latency, _ = execute_explain_analyze(cursor, query)
        cursor.close()
        conn.close()
        return {'query': q_name, 'hint': hint_name, 'latency_postgresql_seconds': latency, 'error': None}
    except Exception as e:
        return {'query': q_name, 'hint': hint_name, 'latency_postgresql_seconds': None, 'error': str(e)}

def main():
    queries = load_queries(QUERIES_DIR)
    hints = load_hints(HINTS_FILE)
    tasks = []

    # Prepare tasks: each query with no hint and each available hint.
    for q_name, query in queries.items():
        tasks.append((q_name, query, 'no_hint', None))
        for h_name, conf in hints.items():
            tasks.append((q_name, query, h_name, conf))

    results = []
    # Adjust max_workers according to available VM cores. For example, 52 workers.
    MAX_WORKERS = 52
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_query, q_name, query, hint_name, conf) for (q_name, query, hint_name, conf) in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
