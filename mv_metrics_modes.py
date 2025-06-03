import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle,re
from scipy import stats

with open('/home/shared/MAD-SSA/pca_clinical.pkl', 'rb') as f:
    data = pickle.load(f)

with open('/home/shared/metrics_20241209.json') as f:
    metrics = json.load(f)
flattened_data = []
for key, values in metrics.items():
    number = re.findall(r'\d+', key)[0]
    flat_entry = {'Pat_no': number}  
    for k, v in values.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                flat_entry[f"{k}_{i}"] = item
        else:
            flat_entry[k] = v
    flattened_data.append(flat_entry)

# Create the DataFrame
df = pd.DataFrame(flattened_data)
df['Pat_no'] = df['Pat_no'].astype(int) 
df.set_index('Pat_no', inplace=True)

# Merge the two DataFrames
mv_data = df.merge(data, on="Pat_no", how="inner")
# print(merged_data)

arc_length_total =mv_data["arc_length_total"]
diameter =mv_data["diameter"]
angle_apex_tilt =mv_data["angle_apex_tilt"]
tent_area =mv_data["tent_area"]
tortuosity =mv_data["tortuosity"]
M7 =data["M7"]
M5 =data["M5"]
M2 =data["M2"]
M1 = mv_data["M1"]
CMR_MA_diam_sys =data["CMR_MA_diam_sys"]
CMR_LGE_POST_Pap_muscle =mv_data["CMR_LGE_POST_Pap_muscle"]
CMR_LGE_ANT_Pap_muscle =mv_data["CMR_LGE_ANT_Pap_muscle"]
CMR_LGE_PM_Y_N =data["CMR_LGE_PM_Y_N"]
CMR_LGE_Myo_adjacent_ml =data["CMR_LGE_Myo_adjacent_ml"]
CMR_LGE_Myocardium_Y_N =data["CMR_LGE_Myocardium_Y_N"]
CMR_MAD_3_CH =data["CMR_MAD_3_CH"]
CMR_MAD_3_CH_Y_N =mv_data["CMR_MAD_3_CH_Y_N"]
Bileaflet_new =data["Bileaflet_new"]
Lateral_ECV =mv_data["Lateral_ECV"]
arr = mv_data["arrhythmic_composite"]
mvp = mv_data["MVP_new"]
# drop patients with nsVT

# Create a pairplot
# sns.pairplot(mv_data, hue="CMR_LGE_POST_Pap_muscle", vars=["angle_apex_tilt","arc_length_total","tent_area","M5","M7","tortuosity","diameter"])
# plt.savefig('figures/pairplot.png')
# plt.show()
# plt.scatter(tent_area, M7, c=mv_data["CMR_LGE_Myocardium_Y_N"], cmap="coolwarm", alpha=0.5)
y_feat = M7
x_feat = CMR_MAD_3_CH
# plt.scatter(x_feat, y_feat, c= mvp,alpha=0.5)

# # plt.xlabel("diamet\er ")
# plt.xlabel(f"{x_feat.name}")
# plt.ylabel(f"{y_feat.name}")
def calculate_mann_whitney(pca_event, pca_noevent, modes):
    """Calculates Mann-Whitney U test p-values for given modes."""
    return [stats.mannwhitneyu(pca_event[mode], pca_noevent[mode])[1] for mode in modes]

pca_clinical_event = mv_data[mv_data["arrhythmic_composite"]]
print(len(pca_clinical_event))
pca_clinical_noevent = mv_data[~mv_data["any_arrhythmia"]]
print(len(pca_clinical_noevent))

ANALYSIS_MODES= ["angle_apex_tilt","arc_length_total","tent_area","tortuosity","diameter"] # [f"M{i}" for i in range(1,  8)]


mann_whitney_p_values = calculate_mann_whitney(
    pca_clinical_event, pca_clinical_noevent, ANALYSIS_MODES
)
print(pd.DataFrame([mann_whitney_p_values], columns=ANALYSIS_MODES, index=["p-value"]))

plt.rcParams.update({"font.size": 14})
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.boxplot(
#     data=mv_data,
#     x = "CMR_MAD_3_CH",
#     y="M5",
#     hue="arrhythmic_composite",
#     ax=ax,
#     palette="Set2",
# )


# ax.set_ylim(-3, 5)

# ax.set_xlabel("Patient Mode Score (Standardized)")
# ax.set_ylabel("PCA Score")
# print(CMR_MA_diam_sys)

# plt.scatter(CMR_MA_diam_sys, M5, alpha=0.5)
# # plt.xlim(0.1,18)
# plt.xlabel("CMR_MA_diam_sys")
# plt.ylabel("M5")
from scipy.stats import pearsonr 
x = CMR_MA_diam_sys
y = M5

valid_indices = ~np.isnan(x) & ~np.isnan(y)
x = x[valid_indices]
y = y[valid_indices]

r, p = pearsonr(x, y)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data points', color='blue', alpha=0.7)

sns.regplot(x=x, y=y, ci=95, scatter_kws={'color': 'blue', 'alpha': 0.7}, 
            line_kws={'color': 'red', 'label': f'r = {r:.2f}, p = {p:.2e}'})
plt.legend(loc='best')
plt.xlabel("CMR_MA_diam_sys")
plt.ylabel("M5")
plt.savefig('figures/m5-diameter.png')