import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import psutil
import os
import subprocess
import time
import datetime
import platform

# Directory constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

def install_dependencies():
    """Installs the necessary dependencies and sets up the environment."""
    print("Installing dependencies...")

    # Create directories
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if not os.path.exists(NOTEBOOKS_DIR):
        os.mkdir(NOTEBOOKS_DIR)
    if not os.path.exists(DATASETS_DIR):
        os.mkdir(DATASETS_DIR)

    # Install dependencies from requirements.txt
    subprocess.run(["pip3", "install", "-r", os.path.join(BASE_DIR, "requirements.txt")])

    # Install fio
    print("Installing fio...")
    subprocess.run(["sudo", "apt-get", "update"])
    subprocess.run(["sudo", "apt-get", "install", "-y", "fio"])

    # Install pyperformance
    subprocess.run(["python3", "-m", "pip", "install", "pyperformance"])

    # Read and use Kaggle credentials
    with open(os.path.join(BASE_DIR, "Secrets.txt"), "r") as f:
        secrets = dict(line.strip().split('=') for line in f if line.strip())

    os.environ['KAGGLE_USERNAME'] = secrets['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY'] = secrets['KAGGLE_KEY']

    # Read benchmark plan to download notebooks and datasets
    with open(os.path.join(BASE_DIR, "benchmark_plan.txt"), "r") as f:
        benchmarks = [line.strip().split(',') for line in f if line.strip()]

    for benchmark in benchmarks:
        notebook_path, b_type, datasets, waittime, num_exec = benchmark

        if b_type == "kaggle_notebook":

            # Download the datasets
            dataset_list = datasets.split(';')
            for dataset in dataset_list:
                dataset_name = dataset.split('/')[-1]  # Extract dataset's name
                print(f"Downloading Kaggle dataset: {dataset_name}")
                subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", DATASETS_DIR])
                dataset_zip = os.path.join(DATASETS_DIR, dataset_name + ".zip")

                # Unzip the dataset
                subprocess.run(["unzip", "-o", dataset_zip, "-d", os.path.join(DATASETS_DIR, dataset_name)])

                # Delete the zip file to save space
                if os.path.exists(dataset_zip):
                    print(f"Deleting Raw Zip File: {dataset_name}.zip")
                    os.remove(dataset_zip)



def get_system_info():
    info = {}

    # Basic platform info
    info['System'] = platform.system()
    info['Node Name'] = platform.node()
    info['Release'] = platform.release()
    info['Version'] = platform.version()
    info['Machine'] = platform.machine()
    info['Processor'] = platform.processor()

    # CPU info
    info['Physical Cores'] = psutil.cpu_count(logical=False)
    info['Total Cores'] = psutil.cpu_count(logical=True)
    info['Max Frequency (MHz)'] = psutil.cpu_freq().max

    # Memory info
    virtual_mem = psutil.virtual_memory()
    info['Total Memory (GB)'] = round(virtual_mem.total / (1024 ** 3), 2)
    info['Available Memory (GB)'] = round(virtual_mem.available / (1024 ** 3), 2)
    info['Memory Usage (%)'] = virtual_mem.percent

    return info


def run_fio_test(input_name, block_size, test_type):
    """Runs an fio test with the given block size and test type."""
    print(f"Running fio test: {block_size}, {test_type}")

    if 'sequential' in test_type:
        io_pattern = 'write' if 'write' in test_type else 'read'
    else:
        io_pattern = 'randwrite' if 'write' in test_type else 'randread'

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{input_name}_{timestamp}_fio_{block_size}_{test_type}.txt"
    result_path = os.path.join(RESULTS_DIR, result_file)

    with open(result_path, "w") as f:
        subprocess.run([
            "fio",
            "--name=fio_test",
            f"--size=1G",
            f"--bs={block_size}",
            f"--rw={io_pattern}",
            "--numjobs=1",
            "--time_based",
            "--runtime=60",
            "--group_reporting",
            "--output-format=json"
        ], stdout=f)

def run_benchmarks(input_name):
    """Runs the benchmarks specified in the benchmark_plan.txt file."""
    print("Starting benchmarks...")

    # Read benchmark plan
    with open(os.path.join(BASE_DIR, "benchmark_plan.txt"), "r") as f:
        benchmarks = [line.strip().split(',') for line in f if line.strip()]

    # Skip the header (assuming the first line is the header)
    benchmarks = benchmarks[1:]

    for benchmark in benchmarks:
        path, b_type, datasets, waittime, num_exec = benchmark
        waittime = int(waittime)
        num_exec = int(num_exec)

        for i in range(num_exec):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"{input_name}_{timestamp}_{os.path.basename(path)}.txt"
            result_path = os.path.join(RESULTS_DIR, result_file)

            if b_type == "standard":
                print(f"Running {path} benchmark...")

                # Check for system info benchmark
                if path == "systeminfo":
                    print("Collecting system information...")
                    system_info = get_system_info()

                    with open(result_path, "w") as f:
                        f.write("System Information:\n")
                        for key, value in system_info.items():
                            f.write(f"{key}: {value}\n")
                    print(f"System information written to {result_file}")

                # Check for fio benchmarks
                elif 'fio' in path:
                    block_size, test_type = path.split('_')[1:3]
                    run_fio_test(input_name, block_size, test_type)

                # Check for pyperformance benchmarks
                elif "pyperformance" in path:
                    benchmark_group = path.split()[1]
                    benchmarks = ""

                    # Define the benchmarks for each group
                    if benchmark_group == "default":
                        benchmarks = "2to3,async_generators,async_tree,async_tree_cpu_io_mixed," \
                                     "async_tree_cpu_io_mixed_tg,async_tree_eager,async_tree_eager_cpu_io_mixed," \
                                     "async_tree_eager_cpu_io_mixed_tg,async_tree_eager_io,async_tree_eager_io_tg," \
                                     "async_tree_eager_memoization,async_tree_eager_memoization_tg,async_tree_eager_tg," \
                                     "async_tree_io,async_tree_io_tg,async_tree_memoization,async_tree_memoization_tg," \
                                     "asyncio_tcp,asyncio_tcp_ssl,asyncio_websockets,chameleon,chaos,comprehensions," \
                                     "concurrent_imap,coroutines,coverage,crypto_pyaes,dask,deepcopy,deltablue," \
                                     "django_template,docutils,dulwich_log,fannkuch,float,gc_collect,gc_traversal," \
                                     "generators,genshi,go,hexiom,html5lib,json_dumps,json_loads,logging,mako,mdp," \
                                     "meteor_contest,nbody,nqueens,pathlib,pickle,pickle_dict,pickle_list," \
                                     "pickle_pure_python,pidigits,pprint,pyflate,python_startup,python_startup_no_site," \
                                     "raytrace,regex_compile,regex_dna,regex_effbot,regex_v8,richards,richards_super," \
                                     "scimark,spectral_norm,sqlalchemy_declarative,sqlalchemy_imperative,sqlglot," \
                                     "sqlglot_optimize,sqlglot_parse,sqlglot_transpile,sqlite_synth,sympy,telco," \
                                     "tomli_loads,tornado_http,typing_runtime_protocols,unpack_sequence,unpickle," \
                                     "unpickle_list,unpickle_pure_python,xml_etree"

                    elif benchmark_group == "math":
                        benchmarks = "float,nbody,pidigits"

                    elif benchmark_group == "regex":
                        benchmarks = "regex_compile,regex_dna,regex_effbot,regex_v8"

                    elif benchmark_group == "serialize":
                        benchmarks = "json_dumps,json_loads,pickle,pickle_dict,pickle_list,pickle_pure_python," \
                                     "tomli_loads,unpickle,unpickle_list,unpickle_pure_python,xml_etree"

                    if benchmarks:
                        print(f"Running specified {benchmark_group} pyperformance benchmarks...")
                        with open(result_path, "w") as f:
                            subprocess.run(
                                ["python3", "-m", "pyperformance", "run", "-b", benchmarks],
                                stdout=f
                            )
                    else:
                        print(f"No benchmarks found for group: {benchmark_group}")


            elif b_type == "kaggle_notebook":
                print(f"Executing Kaggle notebook: {path}")
                
                # Start the timer
                start_time = time.time()

                # Load and execute the notebook
                notebook_path = os.path.join(NOTEBOOKS_DIR, path)
                with open(notebook_path) as f:
                    notebook = nbformat.read(f, as_version=4)

                ep = ExecutePreprocessor(timeout=6000, kernel_name='python3')
                ep.preprocess(notebook, {'metadata': {'path': './notebooks/'}})

                # End the timer and calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time

                # Save execution time to result file
                with open(result_path, "w") as f:
                    f.write(f'Notebook execution time: {execution_time} seconds\n')
            
            # Wait between executions if needed
            time.sleep(waittime)

def parse_and_display_results():
    """Parses benchmark results and prepares them for display."""
    # This function will be developed in stage 3
    pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument('-init', action='store_true', help="Initialize by installing dependencies and setting up environment")
    parser.add_argument('-bench', action='store_true', help="Run benchmarks as per the benchmark plan")
    parser.add_argument('-render', action='store_true', help="Render benchmark results")
    parser.add_argument('--name', type=str, help="Input name for benchmarks", default="benchmark")
    parser.add_argument('--svg', action='store_true', help="Optional flag to output graphs in SVG format")

    args = parser.parse_args()

    if args.init:
        install_dependencies()

    if args.bench:
        input_name = args.name
        run_benchmarks(input_name)

    if args.render:
        # Pass the svg flag to the render script if present
        command = ['python3', 'render_graphs.py']
        if args.svg:
            command.append('--svg')
        subprocess.run(command)
