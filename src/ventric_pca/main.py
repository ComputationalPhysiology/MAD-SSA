import subprocess
def run_script(script_name):
    try:
        subprocess.run(["python3", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")

if __name__ == "__main__":
    scripts = ["main_pc.py","alignment.py", "pca.py"]
    for script in scripts:
        run_script(script)