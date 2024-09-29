import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
from glob import glob

# Directory constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
GRAPHS_DIR = os.path.join(BASE_DIR, 'Graphs')


def ensure_graphs_dir():
    """Ensure the Graphs directory exists."""
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)


def extract_bandwidth_from_fio_output(fio_json_str):
    """Extracts FIO benchmark bandwidth results from a FIO JSON-formatted output."""
    try:
        fio_data = json.loads(fio_json_str)
        jobs = fio_data.get("jobs", [])

        # Initialize empty results for the final output
        output_results = {}

        for job in jobs:
            read_data = job.get("read", {})
            write_data = job.get("write", {})
            job_options = job.get("job options", {})

            # Extract the block size from job options
            block_size = job_options.get("bs", "").lower()

            # Extract the read/write type from job options
            rw_type = job_options.get("rw", "").lower()

            # Initialize the operation type (sequential/random read/write) in output_results if not already present
            if rw_type == "randread":
                op_type = "random_read"
            elif rw_type == "randwrite":
                op_type = "random_write"
            elif rw_type == "read":
                op_type = "sequential_read"
            elif rw_type == "write":
                op_type = "sequential_write"
            else:
                continue  # Skip unsupported types

            # Initialize the operation type in output_results if not already present
            if op_type not in output_results:
                output_results[op_type] = {}

            # Initialize block size in operation type if not already present
            if block_size not in output_results[op_type]:
                output_results[op_type][block_size] = {"throughput": [], "iops": [], "latency": []}

            # Populate the performance metrics
            if "read" in rw_type:
                output_results[op_type][block_size]["throughput"].append(read_data.get("bw", 0) / 1024)  # MB/s
                output_results[op_type][block_size]["iops"].append(read_data.get("iops", 0))
                output_results[op_type][block_size]["latency"].append(read_data.get("clat_ns", {}).get("mean", 0) / 1e6)  # ms
            else:
                output_results[op_type][block_size]["throughput"].append(write_data.get("bw", 0) / 1024)  # MB/s
                output_results[op_type][block_size]["iops"].append(write_data.get("iops", 0))
                output_results[op_type][block_size]["latency"].append(write_data.get("clat_ns", {}).get("mean", 0) / 1e6)  # ms

        return output_results
    except json.JSONDecodeError:
        print("Failed to decode JSON.")
        return None

def extract_pyperformance(log):
    """Extract pyperformance benchmark results."""
    benchmark_results = {}
    # Improved regex using "###" as delimiter for benchmarks
    benchmarks = re.findall(r'###\s*(.*?)\s*###\nMean \+- std dev: ([\d.]+) (\S*) \+- ([\d.]+) (\S*)', log)

    for benchmark in benchmarks:
        name, mean, measurement, stddev, measurement2 = benchmark
        name = "pyperformance " + name
        if measurement == "sec":
            mean = float(mean) * 1000
        if measurement2 == "sec":
            stddev = float(stddev) * 1000
        benchmark_results[name] = {
            "mean": float(mean),
            "stddev": float(stddev)
        }
    return benchmark_results

def parse_fio_result(file_path, group_results):
    """Parses FIO benchmark results."""
    # Extract the block size and type (sequential/random, read/write) from the filename
    base_name = os.path.basename(file_path)
    match = re.search(r'(\d+[kmgKMG])_(sequential|random)(read|write)', base_name)

    if not match:
        print(f"Filename pattern not matched: {base_name}")
        return

    block_size = match.group(1).lower()  # e.g., 1m, 64k
    test_type = match.group(2) + match.group(3)  # e.g., sequentialread, randomwrite

    # Read and process the file to extract relevant performance data
    try:
        with open(file_path, 'r') as f:
            fio_output = f.read()
            bandwidth_data = extract_bandwidth_from_fio_output(fio_output)  # Extract bandwidth data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Save the result under the test type in the 'fio' subgroup, with block size as a direct key
    if bandwidth_data:
        fio_results = group_results.setdefault('fio', {})
        for operation_type, data_by_block_size in bandwidth_data.items():
            if operation_type not in fio_results:
                fio_results[operation_type] = {}

            for block_size_key, metrics in data_by_block_size.items():
                # Directly assign metrics under operation type, grouping by block size
                fio_results[operation_type][block_size_key] = metrics

def parse_pyperformance_result(file_path, group_results):
    """Parses pyperformance benchmark results."""
    with open(file_path, 'r') as f:
        pyperf_output = f.read()
        pyperf_data = extract_pyperformance(pyperf_output)

        # Append results for each benchmark in the 'pybenchmark' subgroup
        group_results.setdefault('pybenchmark', {})
        for test_name, test_data in pyperf_data.items():
            if test_name not in group_results['pybenchmark']:
                group_results['pybenchmark'][test_name] = {"mean": [], "stddev": []}
            group_results['pybenchmark'][test_name]["mean"].append(test_data["mean"])
            group_results['pybenchmark'][test_name]["stddev"].append(test_data["stddev"])

def parse_kaggle_notebook_result(file_path, group_results):
    """Parses the Kaggle notebook execution time from the result."""
    with open(file_path, 'r') as f:
        match = re.search(r'Notebook execution time:\s+(\d+.\d+)\s+seconds', f.read())
        if match:
            notebook_name = os.path.basename(file_path).split('_')[3][:-4]
            exec_time = float(match.group(1))
            # Save the results in the 'notebooks' subgroup
            group_results.setdefault('notebooks', {})
            if notebook_name not in group_results['notebooks']:
                group_results['notebooks'][notebook_name] = []
            group_results['notebooks'][notebook_name].append(exec_time)

def parse_results():
    """Parses the .txt output files in the Results folder and clusters them."""
    files = glob(os.path.join(RESULTS_DIR, "*.txt"))

    grouped_results = {}

    # Regular expression to capture the machine name up to an underscore followed by 8 digits
    machine_name_pattern = re.compile(r'^(.+?)_\d{8}')

    for file in files:
        file_name = os.path.basename(file)

        # If the filename starts with 'benchmark_', assign 'UNKNOWN_MACHINE'
        if file_name.startswith("benchmark_"):
            machine_name = 'UNKNOWN_MACHINE'
        else:
            # Extract machine name from the filename using the regex
            machine_match = machine_name_pattern.match(file_name)
            machine_name = machine_match.group(1).upper() if machine_match else 'UNKNOWN_MACHINE'

        # Ensure there's an entry for this machine
        if machine_name not in grouped_results:
            grouped_results[machine_name] = {}

        # Parse the file based on its benchmark type (e.g., fio, pyperformance, or notebook)
        if 'fio' in file_name:
            parse_fio_result(file, grouped_results[machine_name])
        elif 'pyperformance' in file_name:
            parse_pyperformance_result(file, grouped_results[machine_name])
        elif file_name.endswith('.ipynb.txt'):
            parse_kaggle_notebook_result(file, grouped_results[machine_name])

    return grouped_results


def plot_barchart_with_errorbars(machine_names, means, stddevs, title, ylabel, output_file, colors=None):
    """Plots a bar chart with error bars, optionally with color coding."""
    fig, ax = plt.subplots()
    x = np.arange(len(machine_names))  # the label locations
    bars = ax.bar(x, means, yerr=stddevs, capsize=5, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(machine_names, rotation=45, ha="right")

    # Add the values above the bars, adjusted to account for the error bars
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center the text
            bar.get_height() + stddevs[i],  # Position above the error bar
            f'{means[i]:.2f}',  # Main value (mean)
            ha='center', va='bottom', fontsize=10
        )
        
        # Add the ± standard deviation value just below the main value
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center the text
            bar.get_height() - stddevs[i] - 0.05 * max(means),  # Slightly below the main value
            f'± {stddevs[i]:.2f}',  # Standard deviation
            ha='center', va='top', fontsize=10, color='white'  # Slightly smaller font, gray color
        )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_barchart_without_errorbars(machine_names, means, title, ylabel, output_file, colors=None, operation_types=None, color_dict=None, legend_added=False):
    """Plots a bar chart without error bars, optionally with color coding."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size for more space
    x = np.arange(len(machine_names))  # the label locations
    bars = ax.bar(x, means, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(machine_names, rotation=45, ha="right")

    # Add the values above the bars with formatting conditions
    for i, bar in enumerate(bars):
        mean_value = means[i]
        
        # Determine the font size and formatting based on the mean value
        if mean_value > 9999.99:
            font_size = 8  # Reduce font size by 2
        else:
            font_size = 10  # Default font size

        # Format mean value based on its size
        if mean_value > 999.99:
            text_value = f'{mean_value:.1f}'  # One decimal place
        else:
            text_value = f'{mean_value:.2f}'  # Two decimal places
        
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center the text
            bar.get_height(),  # Position above the bar
            text_value,  # Main value
            ha='center', va='bottom', fontsize=font_size  # Adjusted font size
        )

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.9)  # Adjust the right margin

    # Add a legend for the operation types, if it hasn't been added yet
    if operation_types and not legend_added:
        unique_op_types = list(set(operation_types))
        # Create a list of unique handles for the legend
        unique_handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[op]) for op in unique_op_types]
        ax.legend(unique_handles, unique_op_types, title="Operation Types", loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_execution_times(machine_name, notebook_name, exec_times, svg_flag):
    """Plots execution times with error bars for notebooks."""
    median_time = np.median(exec_times)
    stddev_time = np.std(exec_times)
    file_format = 'svg' if svg_flag else 'png'
    output_file = os.path.join(GRAPHS_DIR, f"{machine_name}_{notebook_name}.{file_format}")
    plot_barchart_with_errorbars([machine_name], [median_time], [stddev_time],
                                 f"Execution Time: {notebook_name}",
                                 'Time (seconds)', output_file)

def plot_pybenchmarks(machine_name, pybench_results, svg_flag):
    """Plots pyperformance benchmark results with mean and std deviation."""
    for benchmark_name, metrics in pybench_results.items():
        means = metrics['mean']
        stddevs = metrics['stddev']
        file_format = 'svg' if svg_flag else 'png'
        output_file = os.path.join(GRAPHS_DIR, f"{machine_name}_pybenchmark_{benchmark_name}.{file_format}")
        plot_barchart_with_errorbars([machine_name], means, stddevs,
                                     f"PyPerformance {benchmark_name} Benchmark",
                                     'Execution Time (ms)', output_file)

def group_machines_by_type(grouped_results):
    """Groups machines by type (AMD/SEV or Intel/TDX)."""
    group1 = {}  # AMD/SEV
    group2 = {}  # Intel/TDX
    group_other = {}  # Neither group
    for machine_name, machine_results in grouped_results.items():
        if 'AMD' in machine_name or 'SEV' in machine_name:
            group1[machine_name] = machine_results
        elif 'intel' in machine_name.lower() or 'TDX' in machine_name:
            group2[machine_name] = machine_results
        else:
            group_other[machine_name] = machine_results
    return group1, group2, group_other

def plot_grouped_results(group_results, group_name, svg_flag, plot_function):
    """Generates comparison charts for a group (either Group1, Group2, or All)."""
    if not group_results:
        print(f"No results to render for {group_name}. Skipping.")
        return
    
    plot_function(group_results, group_name, svg_flag)

def plot_pyperformance_grouped(grouped_results, group_name, svg_flag):
    """Plots pyperformance results for all machines, Group1, and Group2."""
    benchmarks = {}
    # Gather all benchmarks across machines
    for machine_name, machine_results in grouped_results.items():
        for benchmark_name, metrics in machine_results.get('pybenchmark', {}).items():
            if benchmark_name not in benchmarks:
                benchmarks[benchmark_name] = {'mean': [], 'stddev': [], 'machines': []}
            benchmarks[benchmark_name]['mean'].append(np.mean(metrics['mean']))
            benchmarks[benchmark_name]['stddev'].append(np.mean(metrics['stddev']))
            benchmarks[benchmark_name]['machines'].append(machine_name)

    # Plot each benchmark across the groups
    for benchmark_name, data in benchmarks.items():
        file_format = 'svg' if svg_flag else 'png'
        output_file = os.path.join(GRAPHS_DIR, f"pyperf_{benchmark_name}_{group_name}.{file_format}")
        plot_barchart_with_errorbars(data['machines'], data['mean'], data['stddev'],
                                     f"PyPerformance {benchmark_name[14:]} ({group_name})", 
                                     "Execution Time (ms)", output_file)
    """Plots FIO results for all machines, Group1, and Group2, broken down by throughput, latency, and IOPS."""
    fio_metrics = ['throughput', 'latency', 'iops']
    
    # Initialize dictionary to store the data to plot
    fio_data = {metric: {} for metric in fio_metrics}

    for machine_name, machine_results in grouped_results.items():
        for operation_type, block_sizes in machine_results.get('fio', {}).items():
            for block_size, metrics in block_sizes.items():
                for metric in fio_metrics:
                    if block_size not in fio_data[metric]:
                        fio_data[metric][block_size] = {'values': [], 'machines': []}
                    fio_data[metric][block_size]['values'].append(np.mean(metrics[metric]))
                    fio_data[metric][block_size]['machines'].append(machine_name)
    
    # Plot each metric across all block sizes
    for metric, block_data in fio_data.items():
        for block_size, data in block_data.items():
            file_format = 'svg' if svg_flag else 'png'
            output_file = os.path.join(GRAPHS_DIR, f"fio_{metric}_{block_size}_{group_name}.{file_format}")
            plot_barchart_with_errorbars(data['machines'], data['values'], [np.std(data['values'])] * len(data['machines']),
                                         f"FIO {metric.capitalize()} ({block_size}) ({group_name})", 
                                         f"{metric.capitalize()} (MB/s)", output_file)

def plot_barchart(machine_names, means, title, ylabel, output_file):
    """Plots a bar chart without error bars."""
    fig, ax = plt.subplots()
    x = np.arange(len(machine_names))  # the label locations
    bars = ax.bar(x, means)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(machine_names, rotation=45, ha="right")

    # Add the values above the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                f'{means[i]:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_fio_benchmarks(machine_name, fio_results, svg_flag):
    """Plots FIO benchmark results, separating throughput, latency, and IOPS."""
    fio_metrics = ['throughput', 'latency', 'iops']
    
    # Aggregate results by metric, not by block size
    for metric in fio_metrics:
        group_data = {}  # To aggregate data by block size and metric
        for op_type, data in fio_results.items():
            for block_size, metrics in data.items():
                if metric in metrics:
                    if block_size not in group_data:
                        group_data[block_size] = []
                    group_data[block_size].append(metrics[metric][0])  # Only mean values
        
        # For each metric, plot all block sizes in a single chart
        all_block_sizes = list(group_data.keys())
        all_values = [np.mean(group_data[block_size]) for block_size in all_block_sizes]  # Average for each block size
        
        file_format = 'svg' if svg_flag else 'png'
        output_file = os.path.join(GRAPHS_DIR, f"{machine_name}_fio_{metric}_all_sizes.{file_format}")
        plot_barchart(all_block_sizes, all_values,
                      f"FIO {metric.capitalize()} Across Block Sizes",
                      f"{metric.capitalize()} (MB/s)", output_file)

def plot_fio_grouped(grouped_results, group_name, svg_flag):
    """Plots FIO results for all machines, grouped by throughput, latency, and IOPS, and color-coded by operation type."""
    fio_metrics = ['throughput', 'latency', 'iops']

    # Update the list to include all expected operation types
    operation_types = ['read', 'write', 'randomread', 'randomwrite', 'sequentialread', 'sequentialwrite']

    # Proper usage of `get_cmap` to avoid deprecation warnings
    color_map = cm.get_cmap('tab10')
    colors = color_map(np.linspace(0, 1, len(operation_types)))
    color_dict = dict(zip(operation_types, colors))

    # Initialize dictionary to store aggregated data across machines
    fio_data = {metric: {} for metric in fio_metrics}

    for machine_name, machine_results in grouped_results.items():
        for operation_type, block_sizes in machine_results.get('fio', {}).items():
            # Normalize operation type: remove underscores, convert to lowercase
            normalized_op_type = operation_type.replace('_', '').lower()

            if normalized_op_type not in color_dict:
                print(f"Warning: Unrecognized operation type '{normalized_op_type}' encountered. Skipping.")
                continue

            for block_size, metrics in block_sizes.items():
                for metric in fio_metrics:
                    if block_size not in fio_data[metric]:
                        fio_data[metric][block_size] = {'values': [], 'machines': [], 'operation_type': []}
                    fio_data[metric][block_size]['values'].append(np.mean(metrics[metric]))
                    fio_data[metric][block_size]['machines'].append(machine_name)
                    fio_data[metric][block_size]['operation_type'].append(normalized_op_type)

    # Define the desired order for operation types
    operation_order = ['randomread', 'randomwrite', 'sequentialread', 'sequentialwrite']

    # Plot each metric across all block sizes
    for metric, block_data in fio_data.items():
        # Sort block sizes (assuming they are strings, you might need to adjust for int if necessary)
        sorted_block_sizes = sorted(block_data.keys())
        
        for block_size in sorted_block_sizes:
            data = block_data[block_size]
            
            # Sort operation types according to the defined order
            sorted_indices = sorted(range(len(data['operation_type'])), 
                                    key=lambda i: operation_order.index(data['operation_type'][i]))
            
            # Reorder the data based on sorted indices
            sorted_machines = [data['machines'][i] for i in sorted_indices]
            sorted_values = [data['values'][i] for i in sorted_indices]
            sorted_op_types = [data['operation_type'][i] for i in sorted_indices]
            
            # Assign colors based on the sorted operation types
            op_colors = [color_dict[op] for op in sorted_op_types]
            
            # Set the y-axis label based on the metric type
            if metric.lower() == "latency":
                ylabel = "Time (ms)"
            elif metric.lower() == "iops":
                ylabel = "IOPS"  # Empty label for IOPS
            elif metric.lower() == "throughput":
                ylabel = "Throughput (MB/s)"
            else:
                ylabel = "Unknown Value"  # Default label if needed
            file_format = 'svg' if svg_flag else 'png'
            output_file = os.path.join(GRAPHS_DIR, f"fio_{metric}_{block_size}_{group_name}.{file_format}")
            plot_barchart_without_errorbars(
                sorted_machines, 
                sorted_values, 
                f"FIO {metric.capitalize()} ({block_size}) ({group_name})", 
                ylabel,  # Use the dynamic ylabel
                output_file,
                colors=op_colors,  # Pass the colors
                operation_types=sorted_op_types,  # Pass the sorted operation types
                color_dict=color_dict  # Pass the color_dict
            )

def get_shortened_notebook_name(notebook_name):
    """Map full notebook names to shortened, more descriptive names."""
    name_map = {
        "getting-started-with-a-movie-recommendation-system.ipynb": "movie-recommendation",
        "energy-eda-segmentation-and-prediction.ipynb": "energy-eda",
        "credit-fraud-dealing-with-imbalanced-datasets.ipynb": "creditfraud-imbalanced",
        "creditcard-fraud-balance-is-key-feat-pycaret.ipynb": "creditfraud-pycaret",
        "google-play-store-analysis.ipynb": "google-play-analysis",
        "lstm-sentiment-analysis-keras.ipynb": "lstm-sentiment-analysis",
        "e-commerce-eda-purchase-classification.ipynb": "purchase-classification",
        "SimplePrime.ipynb": "simpleprime",
        "twitter-sentiment-analysis.ipynb": "twitter-sentiment",
        "pneumonia-detection-using-cnn-92-6-accuracy.ipynb": "pneumonia-detection",
        "airline-delay-notebook.ipynb": "airline-delay",
    }
    return name_map.get(notebook_name, notebook_name.replace(".ipynb", ""))


def plot_notebook_execution_times_per_machine(grouped_results, svg_flag):
    """Generates graphs for each machine showing all notebook execution times side by side, sorted alphabetically."""
    for machine_name, machine_results in grouped_results.items():
        notebook_names = []
        execution_times = []

        # Gather notebook names and their median execution times
        for notebook_name, exec_times in machine_results.get('notebooks', {}).items():
            shortened_name = get_shortened_notebook_name(notebook_name)
            median_time = np.median(exec_times)
            notebook_names.append(shortened_name)
            execution_times.append(median_time)

        if not notebook_names:
            continue  # Skip if no notebooks are available for this machine

        # Sort notebook names alphabetically, and reorder execution times accordingly
        sorted_notebooks_and_times = sorted(zip(notebook_names, execution_times), key=lambda x: x[0])
        sorted_notebook_names, sorted_execution_times = zip(*sorted_notebooks_and_times)

        # Create a bar chart for the current machine
        file_format = 'svg' if svg_flag else 'png'
        output_file = os.path.join(GRAPHS_DIR, f"{machine_name}_all_notebook_exec_times.{file_format}")

        plot_barchart(sorted_notebook_names, sorted_execution_times, 
                      f"{machine_name} - All Kaggle Notebook Execution Times", 
                      "Time (seconds)", output_file)


def plot_execution_times_grouped(grouped_results, group_name, svg_flag):
    """Plots notebook execution times for all machines in a group, grouped by notebook."""
    for notebook_name in next(iter(grouped_results.values()))['notebooks'].keys():
        shortened_name = get_shortened_notebook_name(notebook_name)
        notebook_exec_times = {}

        # Gather execution times for each machine for this specific notebook
        for machine_name, machine_results in grouped_results.items():
            exec_times = machine_results['notebooks'].get(notebook_name, [])
            if exec_times:
                median_time = np.median(exec_times)
                notebook_exec_times[machine_name] = median_time

        # If no data exists for this notebook in the current group, skip
        if not notebook_exec_times:
            continue

        # Prepare the data for plotting
        machines = list(notebook_exec_times.keys())
        times = list(notebook_exec_times.values())

        file_format = 'svg' if svg_flag else 'png'
        output_file = os.path.join(GRAPHS_DIR, f"notebook_{shortened_name}_exec_times_{group_name}.{file_format}")
        
        # Plot the data
        plot_barchart(machines, times, 
                      f"Notebook Execution Times ({shortened_name}) - {group_name}",
                      "Time (seconds)", output_file)


def plot_results(grouped_results, svg_flag):
    """Generates comparison graphs for all benchmark results, including notebook execution times."""
    ensure_graphs_dir()

    # Group machines by type
    group1, group2, group_other = group_machines_by_type(grouped_results)

    # Plot PyPerformance results for each group (AMD/SEV, Intel/TDX, and Others)
    print("creating PyPerformance graphs...")
    plot_grouped_results(grouped_results, "All Machines", svg_flag, plot_pyperformance_grouped)
    plot_grouped_results(group1, "Group AMD-SEV", svg_flag, plot_pyperformance_grouped)
    plot_grouped_results(group2, "Group Intel-TDX", svg_flag, plot_pyperformance_grouped)

    # Plot FIO benchmarks with operation type color coding
    print("creating Fio graphs...")
    plot_grouped_results(grouped_results, "All Machines", svg_flag, plot_fio_grouped)
    plot_grouped_results(group1, "Group AMD-SEV", svg_flag, plot_fio_grouped)
    plot_grouped_results(group2, "Group Intel-TDX", svg_flag, plot_fio_grouped)

    # Plot notebook execution times for each group
    print("creating Kaggle Notebook graphs...")
    plot_grouped_results(grouped_results, "All Machines", svg_flag, plot_execution_times_grouped)
    plot_grouped_results(group1, "Group AMD-SEV", svg_flag, plot_execution_times_grouped)
    plot_grouped_results(group2, "Group Intel-TDX", svg_flag, plot_execution_times_grouped)

    # Plot notebook execution times for individual machines
    plot_notebook_execution_times_per_machine(grouped_results, svg_flag)
    plot_notebook_execution_times_per_machine(group1, svg_flag)
    plot_notebook_execution_times_per_machine(group2, svg_flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render benchmark graphs")
    parser.add_argument('--svg', action='store_true', help="Output graphs in SVG format instead of PNG")
    
    print("starting render...")
    args = parser.parse_args()

    print("reading benchmarks data...")
    # Parse the results
    grouped_results = parse_results()
    print("done reading benchmark data.")

    # Plot and save the graphs
    print("creating graphs...")
    plot_results(grouped_results, args.svg)
    print("done creating graphs.")
