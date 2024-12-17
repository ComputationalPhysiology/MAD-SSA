import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import os

# Constants
CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
PCA_SCORES_CSV = "./PCA_Results/pca_scores.csv"
NUM_MODES = 7
ANALYSIS_MODES = [f"M{i}" for i in range(1, NUM_MODES + 1)]


# Functions
def load_patient_data(file_path):
    """Loads patient clinical data from an Excel file."""
    try:
        data = pd.read_excel(file_path).drop([0, 1])
        data["Pat_no"] = data["Pat_no"].astype(int)
        data.set_index("Pat_no", inplace=True)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_pca_scores(file_path):
    """Loads and standardizes PCA scores from a CSV file."""
    try:
        pca_scores = pd.read_csv(file_path).set_index("Pat_no")
        pca_scores.index = pca_scores.index.astype(int)
        pca_scores = (pca_scores - pca_scores.mean()) / pca_scores.std()
        return pca_scores
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def calculate_mann_whitney(pca_event, pca_noevent, modes):
    """Calculates Mann-Whitney U test p-values for given modes."""
    return [stats.mannwhitneyu(pca_event[mode], pca_noevent[mode])[1] for mode in modes]

def annotate_p_values(ax, p_values, y_pos=3, color="black"):
    """Annotates p-values on the plot."""
    for i, p_val in enumerate(p_values):
        ax.annotate(f"P = {p_val:.2f}", (i - 0.4, y_pos), color=color)

def run_analysis(clinical_data_xl, pca_scores_csv, analysis_modes, binary_outcome_column, results_dir):
    clinical_data = load_patient_data(clinical_data_xl)
    pca_scores = load_pca_scores(pca_scores_csv)

    if clinical_data is None or pca_scores is None:
        print("Failed to load data. Exiting.")
        return

    clinical_data[binary_outcome_column] = clinical_data[binary_outcome_column].fillna(0).astype(bool)
    
    pca_clinical = pca_scores.merge(clinical_data, on="Pat_no")
    pca_clinical_event = pca_clinical[pca_clinical[binary_outcome_column]]
    pca_clinical_noevent = pca_clinical[~pca_clinical[binary_outcome_column]]


    # Statistical Analysis
    mann_whitney_p_values = calculate_mann_whitney(pca_clinical_event, pca_clinical_noevent, analysis_modes)
    print(pd.DataFrame([mann_whitney_p_values], columns=analysis_modes, index=["p-value"]))

    # Prepare Data for Plotting
    pca_scores_stacked = pca_scores[analysis_modes].stack().reset_index(level=1)
    pca_scores_stacked.columns = ["mode", "score"]
    clinical_data_pca_stacked = clinical_data.merge(pca_scores_stacked, on="Pat_no")

    # Plotting
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=clinical_data_pca_stacked,
        x="mode",
        y="score",
        hue=binary_outcome_column,
        ax=ax,
        palette="Set2",
    )

    ax.set_ylim(-3, 5)
    annotate_p_values(ax, mann_whitney_p_values, y_pos=4)
    ax.set_xlabel("Patient Mode Score (Standardized)")
    ax.set_ylabel("PCA Score")
    ax.legend(title=binary_outcome_column.replace('_', ' ').title(), loc="lower center")

    plt.tight_layout()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_file_path = os.path.join(results_dir, f"{binary_outcome_column}_mode_comparison.png")
    plt.savefig(plot_file_path)
    plt.show()

FEATURE_LGE = ["LGE_myo_Hopp_ml", "LGE_PM_Hopp_ml", "LGE_chordae_Hopp", "LGE_MV_Hopp", "CMR_LGE_POST_Pap_muscle", "CMR_LGE_ANT_Pap_muscle", "CMR_LGE_PM_Y_N", "CMR_LGE_Myo_adjacent_ml", "CMR_LGE_Myocardium_Y_N"]
FEATURE_MAD = ["CMR_MAD_3_CH","CMR_MAD_3_CH_Y_N", "Post_leaf", "Ant_leaf", "MVP_new","Bileaflet_new"]
results_dir = f"./mode_comparison_results/FEATURE_LGE"
# for i in FEATURE_LGE:
#     run_analysis(CLINICAL_DATA_XL,PCA_SCORES_CSV, ANALYSIS_MODES, i, results_dir)

