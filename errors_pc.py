import pandas as pd

# Load data from the CSV file
data = pd.read_csv('/home/shared/00_data/errors_summary_d.csv')

# Calculate mean and standard deviation for Mean Epi Error
mean_epi = data['Mean Epi Error'].mean()
std_epi = data['Mean Epi Error'].std()

# Calculate mean and standard deviation for Mean Endo Error
mean_endo = data['Mean Endo Error'].mean()
std_endo = data['Mean Endo Error'].std()

print(f"Mean Epi Error: {mean_epi:.4f} ± {std_epi:.4f}")
print(f"Mean Endo Error: {mean_endo:.4f} ± {std_endo:.4f}")