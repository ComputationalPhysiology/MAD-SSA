# import pandas as pd
# import os
# import sys
# import yaml
# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# from scipy import stats

# def load_patient_data(clinical_datapath):
#     patientdata = pd.read_excel(clinical_datapath).drop([0,1])
#     patientdata["Pat_no"] = patientdata["Pat_no"].astype(int)
#     patientdata = patientdata.set_index("Pat_no")
#     return patientdata



# CLINICAL_DATA_XL = "ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
# NUM_MODES = 10
# ANALYSIS_MODES = ["M{}".format(i) for i in range(1, NUM_MODES +1)]
# clinical_data = load_patient_data(CLINICAL_DATA_XL)
# has_event = clinical_data[["Aborted_cardiac_arrest","Ventricular_tachycardia"]].fillna(0).sum(axis = 1) > 0

# any_arrhythmia = clinical_data[["Aborted_cardiac_arrest",
#                                "Ventricular_tachycardia",
#                                "nsVT"]].fillna(0).sum(axis = 1) > 0

# clinical_data["arrhythmic_composite"] = has_event
# clinical_data["any_arrhythmia"] = any_arrhythmia
# mad_arrhytimic_patients = clinical_data[has_event].index
# pca_scores =  pd.read_csv("/home/shared/MAD-SSA/PCA_Results/pca_scores.csv").set_index("Pat_no")
# pca_scores.index = np.array([str(case) for case in pca_scores.index], dtype=int)

# pca_scores.index.rename('Pat_no', inplace=True) 
# pca_scores = (pca_scores - pca_scores.mean())/pca_scores.std()
# pca_clinical = pca_scores.merge(clinical_data, on = "Pat_no")
# pca_clinical_event = pca_clinical.loc[pca_clinical["arrhythmic_composite"]]
# pca_clinical_noevent = pca_clinical.loc[~pca_clinical["any_arrhythmia"]]
# manu_stats = [stats.mannwhitneyu(pca_clinical_event[M], pca_clinical_noevent[M])[1] for M in ANALYSIS_MODES]
# print(pd.DataFrame([manu_stats], columns = ANALYSIS_MODES, index = ["p-value"]))


# pca_scores_stacked = pca_scores[ANALYSIS_MODES].stack()
# pca_scores_stacked = pd.DataFrame(pca_scores_stacked).reset_index(level = [1])
# pca_scores_stacked.columns = ["mode", "score"]

# clinical_data_pca_stacked= clinical_data.merge(pca_scores_stacked, on = "Pat_no")

# clinical_data_pca = clinical_data.merge(pca_scores, on = "Pat_no")
# plt.rcParams.update({'font.size': 14})

# ax = sns.boxplot(data=clinical_data_pca_stacked,
#                  x="mode",
#                  y="score",
#                  hue="arrhythmic_composite")



# plt.ylim(-3, 5)
# plt.annotate("P = {:.2f}".format(manu_stats[0]), (-0.4, 3))
# plt.xlabel("patient mode score (std)")
# # plt.annotate("P = {:.2f}".format(manu_stats[2]), (-0.4, 3),color = "green")
# plt.annotate("P = {:.2f}".format(manu_stats[1]), (0.6, 4))
# plt.annotate("P = {:.2f}".format(manu_stats[2]), (1.6, 3))
# plt.annotate("P = {:.2f}".format(manu_stats[3]), (2.6, 3))
# plt.annotate("P = {:.2f}".format(manu_stats[4]), (3.6, 3))
# # # plt.annotate("P = {:.2f}".format(manu_stats[4]), (3.6, 3))
# #off legend
# plt.legend().set_visible(False)

# #plt.savefig("arrhythmia_mode_comparison.png")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

# Constants
CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
PCA_SCORES_CSV = "/home/shared/MAD-SSA/PCA_Results/pca_scores.csv"
NUM_MODES = 10
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

# Load Data
clinical_data = load_patient_data(CLINICAL_DATA_XL)
pca_scores = load_pca_scores(PCA_SCORES_CSV)

if clinical_data is not None and pca_scores is not None:
    # Process Clinical Data
    clinical_data["arrhythmic_composite"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia"]].fillna(0).sum(axis=1) > 0
    )
    clinical_data["any_arrhythmia"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia", "nsVT"]].fillna(0).sum(axis=1) > 0
    )

    # Merge Clinical and PCA Data
    pca_clinical = pca_scores.merge(clinical_data, on="Pat_no")
    pca_clinical_event = pca_clinical[pca_clinical["arrhythmic_composite"]]
    pca_clinical_noevent = pca_clinical[~pca_clinical["any_arrhythmia"]]

    # Statistical Analysis
    mann_whitney_p_values = calculate_mann_whitney(
        pca_clinical_event, pca_clinical_noevent, ANALYSIS_MODES
    )
    print(pd.DataFrame([mann_whitney_p_values], columns=ANALYSIS_MODES, index=["p-value"]))

    # Prepare Data for Plotting
    pca_scores_stacked = pca_scores[ANALYSIS_MODES].stack().reset_index(level=1)
    pca_scores_stacked.columns = ["mode", "score"]
    clinical_data_pca_stacked = clinical_data.merge(pca_scores_stacked, on="Pat_no")

    # Plotting
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=clinical_data_pca_stacked,
        x="mode",
        y="score",
        hue="arrhythmic_composite",
        ax=ax,
        palette="Set2",
    )


    ax.set_ylim(-3, 5)
    annotate_p_values(ax, mann_whitney_p_values, y_pos=4)
    ax.set_xlabel("Patient Mode Score (Standardized)")
    ax.set_ylabel("PCA Score")
    ax.legend(title="Arrhythmic Composite", loc="lower center")

    plt.tight_layout()
    plt.savefig("arrhythmia_mode_comparison.png")
    plt.show()
