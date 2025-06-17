
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)

def main():
    run_command("saxomode-pc")
    run_command("saxomode-alignment")
    run_command("saxomode-pca")

if __name__ == "__main__":
    main()