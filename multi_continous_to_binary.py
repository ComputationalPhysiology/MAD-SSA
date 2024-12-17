import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import os

# Constants
CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
PCA_SCORES_CSV = "./PCA_Results_final_height/pca_scores.csv"
NUM_MODES = 7
ANALYSIS_MODES = [f"M{i}" for i in [   7]]

# Functions
def load_patient_data(file_path):
    """Load patient clinical data from an Excel file."""
    try:
        data = pd.read_excel(file_path).drop([0, 1])
        data["Pat_no"] = data["Pat_no"].astype(int)
        data.set_index("Pat_no", inplace=True)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_pca_scores(file_path):
    """Load and standardize PCA scores from a CSV file."""
    try:
        pca_scores = pd.read_csv(file_path).set_index("Pat_no")
        pca_scores.index = pca_scores.index.astype(int)
        pca_scores = (pca_scores - pca_scores.mean()) / pca_scores.std()
        return pca_scores
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def run_logistic_regression(X, y):
    """Run logistic regression and return the model results."""
    try:
        logit_model = sm.Logit(y, sm.add_constant(X))
        result = logit_model.fit(disp=False)
        return result
    except Exception as e:
        print(f"Error during logistic regression: {e}")
        return None

def run_analysis(clinical_data_xl, pca_scores_csv, analysis_modes, results_dir):
    # Load data
    clinical_data = load_patient_data(clinical_data_xl)
    pca_scores = load_pca_scores(pca_scores_csv)
    
    if clinical_data is None or pca_scores is None:
        print("Data loading failed. Aborting analysis.")
        return

    # Create binary outcome variables
    clinical_data["arrhythmic_composite"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia"]].fillna(0).sum(axis=1) > 0
    )
    clinical_data["any_arrhythmia"] = (
        clinical_data[["Aborted_cardiac_arrest", "Ventricular_tachycardia", "nsVT"]].fillna(0).sum(axis=1) > 0
    )

    # Merge PCA scores and clinical data
    pca_clinical = pca_scores.merge(clinical_data, left_index=True, right_index=True, how="inner")
    # save as pickle
    
    # Define features (X) and target (y)
    features = analysis_modes + ["CMR_MAD_percentage_ring","CMR_Largest_MAD","CMR_MAD_3_CH"]
    print(f"Running logistic regression for features: {features}")
    X = pca_clinical[features].fillna(0)
    y = pca_clinical["CMR_LGE_Myocardium_Y_N"].fillna(0).astype(int)

    # Run logistic regression
    result = run_logistic_regression(X, y)
    if result:
        print(result.summary())

        # Save results
        result_file = os.path.join(results_dir, "logistic_regression_results.txt")
        with open(result_file, "w") as f:
            f.write(result.summary().as_text())
        print(f"Results saved to {result_file}")

        # Plot p-values
        plt.figure(figsize=(10, 6))
        sns.heatmap(result.pvalues.to_frame().T, annot=True, cmap="coolwarm", cbar=False)
        plt.title("P-values for Logistic Regression Coefficients")
        plt.savefig(os.path.join(results_dir, "logistic_regression_pvalues.png"))
        plt.close()

# Directories
FEATURE_LGE = [
    "LGE_myo_Hopp_ml", "LGE_PM_Hopp_ml", "LGE_chordae_Hopp", "LGE_MV_Hopp", 
    "CMR_LGE_POST_Pap_muscle", "CMR_LGE_ANT_Pap_muscle", "CMR_LGE_PM_Y_N", 
    "CMR_LGE_Myo_adjacent_ml", "CMR_LGE_Myocardium_Y_N"
]
FEATURE_MAD = [
    "CMR_MAD_3_CH", "CMR_MAD_3_CH_Y_N", "Post_leaf", "Ant_leaf", "MVP_new", "Bileaflet_new"
]
results_dir = "./mode_comparison_results/FEATURE_LGE"
os.makedirs(results_dir, exist_ok=True)

# Run analysis
run_analysis(CLINICAL_DATA_XL, PCA_SCORES_CSV, ANALYSIS_MODES, results_dir)

# import pandas as pd
# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import os

# # Constants
# CLINICAL_DATA_XL = "../ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
# PCA_SCORES_CSV = "./PCA_Results_final_height/pca_scores.csv"
# NUM_MODES = 7
# ANALYSIS_MODES = [f"M{i}" for i in [2, 5, 7]]
# ARRHYTHMIA_COLUMNS = ["Aborted_cardiac_arrest", "Ventricular_tachycardia"]

# # Load Data
# def load_patient_data(file_path):
#     data = pd.read_excel(file_path).drop([0, 1])
#     data["Pat_no"] = data["Pat_no"].astype(int)
#     data.set_index("Pat_no", inplace=True)
#     return data

# def load_pca_scores(file_path):
#     pca_scores = pd.read_csv(file_path).set_index("Pat_no")
#     pca_scores.index = pca_scores.index.astype(int)
#     pca_scores = (pca_scores - pca_scores.mean()) / pca_scores.std()
#     return pca_scores

# # Combine arrhythmia columns into a single binary outcome
# def create_combined_outcome(clinical_data):
#     clinical_data['Combined_Arrhythmia'] = clinical_data[ARRHYTHMIA_COLUMNS].fillna(0).max(axis=1)
#     return clinical_data

# # Run Analysis
# def run_elastic_net_analysis(clinical_data_xl, pca_scores_csv, analysis_modes, results_dir):
#     clinical_data = load_patient_data(clinical_data_xl)
#     pca_scores = load_pca_scores(pca_scores_csv)
    
#     clinical_data = create_combined_outcome(clinical_data)
    
#     pca_clinical = pca_scores.merge(clinical_data, on="Pat_no")
#     X = pca_clinical[analysis_modes]
#     y = pca_clinical['Combined_Arrhythmia']

#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#     # Elastic Net Logistic Regression
#     model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000)
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)

#     # Save and print results
#     coef = model.coef_[0]
#     intercept = model.intercept_[0]
#     results = pd.DataFrame({
#         'Feature': ['Intercept'] + analysis_modes,
#         'Coefficient': [intercept] + list(coef)
#     })
#     results_path = os.path.join(results_dir, "elastic_net_coefficients.csv")
#     results.to_csv(results_path, index=False)
#     print(results)
    
#     # Print classification report and confusion matrix
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))

#     # Plotting coefficients
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Feature', y='Coefficient', data=results)
#     plt.title(f'Elastic Net Coefficients')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_dir, 'elastic_net_coefficients.png'))
#     plt.close()

# # Run the analysis
# results_dir = "./mode_comparison_results/Combined_Arrhythmia"
# os.makedirs(results_dir, exist_ok=True)
# run_elastic_net_analysis(CLINICAL_DATA_XL, PCA_SCORES_CSV, ANALYSIS_MODES, results_dir)