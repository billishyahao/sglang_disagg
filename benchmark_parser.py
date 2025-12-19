#!/usr/bin/env python3
"""
Benchmark Log Parser

Extracts performance metrics from SGLang benchmark logs and displays results
in a formatted table. Optionally saves results to CSV file.
"""

import re
import sys
import pandas as pd
import argparse

def parse_benchmark_log(logfile):
    """Parse benchmark log file and extract performance metrics."""
    with open(logfile, 'r') as f:
        content = f.read()

    results = []

    # Split content by benchmark runs (each starts with "Benchmarking on")
    runs = re.split(r'export_file:', content)[1:]  # Skip first empty split

    for run in runs:
        # Extract parameters from the export_file path
        # Example: /sglang_disagg/logs/slurm_job-430/sglang_isl_1024_osl_1024/concurrency_64_req_rate_inf_gpus_24_ctx_8_gen_16
        # match = re.search(r'sglang_isl_(\d+)_osl_(\d+)/concurrency_(\d+)_req_rate_([^_]+)_gpus_(\d+)_ctx_(\d+)_gen_(\d+)', run)
        # if not match:
        #     continue

        # question_len = int(match.group(1))
        # output_len = int(match.group(2))
        # concurrency = int(match.group(3))
        # # req_rate = match.group(4) # Not used in output dict yet
        # # total_gpus = int(match.group(5)) # Not used in output dict yet
        # xp = int(match.group(6))
        # yd = int(match.group(7))
        # model_path = "Unknown" # Model name is not in the line, defaulting


        # Extract dataset statistics
        question_len = None
        output_len = None
        concurrency = None
        request_rate = None
        model_path = None

        # Extract parameters from the export_file path which is at the beginning of the chunk
        # The chunk starts with: /sglang_disagg/logs/...
        match = re.search(r'sglang_isl_(\d+)_osl_(\d+)/concurrency_(\d+)', run)
        if match:
             question_len = int(match.group(1))
             output_len = int(match.group(2))
             concurrency = int(match.group(3))
        
        # Attempt to find model in the command line execution
        model_match = re.search(r'benchmark_serving\.py.+--model\s+([^\s]+)', run, re.DOTALL)
        if model_match:
             model_path = model_match.group(1)

        # Find benchmark result block
        if '============ Serving Benchmark Result ============' in run:
            def extract(pattern):
                m = re.search(pattern, run)
                return float(m.group(1).replace(',', '')) if m else None

            # Extract all metrics
            successful_requests = extract(r'Successful requests:\s+([\d,]+)')
            duration = extract(r'Benchmark duration \(s\):\s+([\d\.]+)')
            req_throughput = extract(r'Request throughput \(req/s\):\s+([\d\.]+)')
            
            # Input token throughput might be missing, calculate if needed
            total_input_tokens = extract(r'Total input tokens:\s+([\d,]+)')
            input_tok_throughput = extract(r'Input token throughput \(tok/s\):\s+([\d,\.]+)')
            if input_tok_throughput is None and total_input_tokens and duration:
                input_tok_throughput = total_input_tokens / duration

            output_tok_throughput = extract(r'Output token throughput \(tok/s\):\s+([\d,\.]+)')
            
            # Case insensitive match for Total Token throughput
            total_tok_throughput = extract(r'Total [Tt]oken throughput \(tok/s\):\s+([\d,\.]+)')
            
            mean_e2e = extract(r'Mean E2EL? \(ms\):\s+([\d,\.]+)') # Handle E2E or E2EL
            mean_ttft = extract(r'Mean TTFT \(ms\):\s+([\d,\.]+)')
            mean_itl = extract(r'Mean ITL \(ms\):\s+([\d,\.]+)')

            results.append({
                'Model': model_path,
                'ISL': question_len,
                'OSL': output_len,
                'Concurrency': concurrency,
                'Request_Throughput_req_s': req_throughput,
                'Input_Token_Throughput_tok_s': input_tok_throughput,
                'Output_Token_Throughput_tok_s': output_tok_throughput,
                'Total_Token_Throughput_tok_s': total_tok_throughput,
                'Mean_E2E_Latency_ms': mean_e2e,
                'Mean_TTFT_ms': mean_ttft,
                'Mean_ITL_ms': mean_itl,
            })

    return results

def format_dataframe(df):
    """Format the dataframe for better readability."""
    # Format numeric columns
    numeric_cols = [
        'Request Throughput (req/s)', 'Input Token Throughput (tok/s)',
        'Output Token Throughput (tok/s)', 'Total Token Throughput (tok/s)',
        'Mean E2E Latency (ms)', 'Mean TTFT (ms)', 'Mean ITL (ms)'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

    # Format integer columns with commas
    integer_cols = ['Total Input Tokens', 'Total Output Tokens']
    for col in integer_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else x)

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Parse SGLang benchmark logs and extract performance metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmark.log                    # Display results on screen
  %(prog)s benchmark.log --csv results.csv # Save to CSV file
  %(prog)s benchmark.log --csv             # Save to auto-named CSV file

The tool extracts metrics including:
  - Model configuration (xP/yD)
  - Input/Output sequence lengths (ISL/OSL)
  - Concurrency levels and prompt counts
  - Token throughput (input/output/total)
  - Latency metrics (E2E, TTFT, ITL)
        """
    )

    # Required arguments
    parser.add_argument(
        'logfile',
        help='Path to the benchmark log file to parse'
    )

    # Optional arguments
    parser.add_argument(
        '--csv',
        nargs='?',
        const='benchmark_results.csv',
        metavar='FILE',
        help='Save results to CSV file. If no filename provided, uses "benchmark_results.csv"'
    )

    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use compact output format (fewer columns)'
    )

    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='Skip screen output, only save to CSV (requires --csv)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.no_screen and not args.csv:
        parser.error("--no-screen requires --csv option")

    try:
        results = parse_benchmark_log(args.logfile)
    except FileNotFoundError:
        print(f"Error: File '{args.logfile}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No benchmark results found in the log file.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(results)

    # Select columns based on compact option
    if args.compact:
        columns = [
            'Model', 'ISL', 'OSL', 'Concurrency',
            'Request Throughput (req/s)', 'Total Token Throughput (tok/s)',
            'Mean E2E Latency (ms)', 'Mean TTFT (ms)', 'Mean ITL (ms)'
        ]
        df = df[columns]

    # Format for display
    display_df = format_dataframe(df.copy())

    # Screen output
    if not args.no_screen:
        print("Benchmark Results Summary:")
        print("=" * 120)
        print(display_df.to_string(index=False))
        print(f"\nTotal runs parsed: {len(results)}")

    # CSV output
    if args.csv:
        try:
            # Save original unformatted data to CSV for better data processing
            df.to_csv(args.csv, index=False)
            if not args.no_screen:
                print(f"\nResults saved to: {args.csv}")
        except Exception as e:
            print(f"Error saving CSV file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
