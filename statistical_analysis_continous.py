
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import os


CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
PCA_SCORES_CSV = "./PCA_Results/pca_scores.csv"
NUM_MODES = 7
ANALYSIS_MODES = [f"M{i}" for i in range(1, NUM_MODES + 1)]
RESULTS_DIR = "/home/shared/MAD-SSA/mode_comparison_results/continuous"


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


def calculate_correlations(clinical_data, pca_scores, continuous_vars, modes):
    """Calculates Pearson correlation coefficients and p-values between continuous variables and PCA modes."""

    aligned_data = clinical_data[continuous_vars].join(pca_scores[modes], how='inner')
    
    # Fill missing values with 0
    aligned_data = aligned_data.fillna(0)
    
    if aligned_data.empty:
        print("No data available after aligning and filling missing values.")
        return pd.DataFrame(index=continuous_vars, columns=modes), pd.DataFrame(index=continuous_vars, columns=modes)
    
    correlations = pd.DataFrame(index=continuous_vars, columns=modes)
    p_values = pd.DataFrame(index=continuous_vars, columns=modes)
    for var in continuous_vars:
        for mode in modes:
            corr, p_val = stats.pearsonr(aligned_data[var], aligned_data[mode])
            correlations.loc[var, mode] = corr
            p_values.loc[var, mode] = p_val
    return correlations, p_values

def plot_correlations(correlations, p_values, output_path_corr, output_path_pval):
    """Plots heatmaps of the correlation coefficients and p-values."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations.astype(float), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Between Continuous Variables and PCA Modes")
    plt.xlabel("PCA Modes")
    plt.ylabel("Continuous Variables")
    plt.tight_layout()
    plt.savefig(output_path_corr)
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(p_values.astype(float), annot=True, cmap='coolwarm', center=0)
    plt.title("P-values for Correlation Between Continuous Variables and PCA Modes")
    plt.xlabel("PCA Modes")
    plt.ylabel("Continuous Variables")
    plt.tight_layout()
    plt.savefig(output_path_pval)
    plt.show()

# Main analysis function
def run_correlation_analysis(clinical_data_xl, pca_scores_csv, analysis_modes, continuous_vars, results_dir):
    clinical_data = load_patient_data(clinical_data_xl)
    pca_scores = load_pca_scores(pca_scores_csv)
    
    correlations,pvalues = calculate_correlations(clinical_data, pca_scores, continuous_vars, analysis_modes)
    print(correlations)

    # Plot and save correlations
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_file_path = os.path.join(results_dir, "correlation_heatmap_mad.png")
    plot_file_path2 = os.path.join(results_dir, "pvalue_mad.png")
    plot_correlations(correlations,pvalues,plot_file_path, plot_file_path2)

continuous_vars = ["CMR_max_ES_thickness_ineferolat_wall_MID", "CMR_max_ES_thickness_ineferolat_wall",
                "CMR_max_ED_thickness_ineferolat_wall_MID", "CMR_max_ED_thickness_ineferolat_wall"]
featurecmr = ["CMR_EF", "CMR_EDV", "CMR_ESV", "CMR_LV_ESV", "CMR_LV_EDV", "CMR_LV_mass", 
              "CMR_MA_diam_dia", "CMR_MA_diam_sys"]
feature_mad = [ "CMR_Degree_largest_MAD", "CMR_Largest_MAD", "CMR_total_degrees_MAD"] 
run_correlation_analysis(CLINICAL_DATA_XL, PCA_SCORES_CSV, ANALYSIS_MODES, feature_mad, RESULTS_DIR)


