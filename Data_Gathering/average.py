import re
import csv

def parse_line(line):
    """
    Parses a line from the log and returns a tuple of (query, hint, latency)
    if the line contains a "Finished query:" entry.
    """
    # Adjusted regex:
    # - \s* to allow for any amount of whitespace
    # - ([^ ]+) is replaced with (\S+) for non-whitespace
    # - Captures latency as a sequence of digits and periods
    pattern = r"Finished query:\s*(\S+)\s+with hint:\s*(\S+)\s+\(latency:\s*([\d\.]+)\)"
    match = re.search(pattern, line)
    if match:
        query = match.group(1)
        hint = match.group(2)
        latency = match.group(3)
        return query, hint, latency
    else:
        # Debug: print lines that did not match the expected format.
        # Uncomment the next line to see non-matching lines.
        # print("No match for line:", line.strip())
        return None

def main():
    input_file = "/Users/raahimlone/rahhh/Data_Gathering/data1.txt"   # Your input file path
    output_file = "output.csv"  # Output CSV file

    parsed_data = []
    with open(input_file, "r") as f:
        for line in f:
            result = parse_line(line)
            if result:
                parsed_data.append(result)

    # Write the parsed data into a CSV file with three columns.
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "hint", "latency_postgresql_seconds"])
        writer.writerows(parsed_data)

    print(f"Processed {len(parsed_data)} query executions. Output saved to {output_file}.")

if __name__ == "__main__":
    main()
