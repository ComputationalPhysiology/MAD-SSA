import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_DIRECTORY = os.path.join(BASE_DIR,"controls", "00_data")
OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "Aligned_Models")
SETTINGS_DIRECTORY = os.path.join(BASE_DIR, "controls", "settings")
ES_files_controls = os.path.join(BASE_DIR, "controls", "ES_files_controls")
PCA_directory = os.path.join(BASE_DIR, "controls", "PCA_Results")