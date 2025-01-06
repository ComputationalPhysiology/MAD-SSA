import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory containing patient data
base_case_dir = Path("/home/shared/00_data")
script_path = Path("/home/shared/MAD-SSA/create_3d_mesh.py")  # Adjust to the path of your existing script

# Function to run the existing script on a specific case
def run_experiment(case_name):
    case_output = base_case_dir / case_name / "00_results"

    # Build the command for the subprocess
    cmd = [
        "python3", script_path.as_posix(),
        "-n", case_name,
        # "-d", base_case_dir.as_posix(),
        "-delauny"
        # "-o", case_output.as_posix(),
        
    ]

    # Execute the script with a timeout
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        logger.info(f"Finished processing case: {case_name}")
        return True  # Indicates success
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while processing case {case_name}")
        return False  # Indicates failure
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing case {case_name}: {e}")
        logger.error(e.stderr.decode())
        return False  # Indicates failure

def get_all_cases(directory):
    if not directory.exists():
        logger.error(f"Directory {directory} does not exist.")
        return []
    return [item.name for item in directory.iterdir() if item.is_dir()]

def main():
    if not base_case_dir.exists():
        logger.error(f"Base case directory {base_case_dir} does not exist.")
        return

    if not script_path.exists():
        logger.error(f"Script {script_path} does not exist.")
        return

    # Get all case names from the base directory
    case_names = get_all_cases(base_case_dir)

    # List to store failed cases
    failed_cases = []

    # Process cases in parallel
    max_workers = min(os.cpu_count() or 4, 10)  # Adjust max_workers as needed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_experiment, case_name): case_name for case_name in case_names}

        for future in as_completed(futures):
            case_name = futures[future]
            try:
                success = future.result()
                if not success:
                    failed_cases.append(case_name)
                    logger.warning(f"Case {case_name} failed to process. Continuing with other cases.")
            except Exception as e:
                logger.error(f"Unexpected error while processing case {case_name}: {e}")
                failed_cases.append(case_name)

    # Log failed cases at the end
    if failed_cases:
        logger.error(f"The following cases failed to process: {', '.join(failed_cases)}")
    else:
        logger.info("All cases processed successfully.")

if __name__ == "__main__":
    main()