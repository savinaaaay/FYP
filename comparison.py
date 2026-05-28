import pandas as pd
import numpy as np

# ============================================================
# 1) LOAD CSV FILE
# ============================================================

file_path = r"D:\Savini\Wen Codes\savini_new_results\Computation_Time_Comparison_t.csv"

df = pd.read_csv(file_path)

print("Columns found:")
print(df.columns.tolist())

# ============================================================
# 2) MAKE SURE NUMERIC COLUMNS ARE NUMERIC
# ============================================================

numeric_cols = [
    "p",
    "nbES",
    "confidence_level",
    "num_scenarios_predicted",
    "total_time_predicted",
    "avg_time_predicted",
    "num_scenarios_original",
    "total_time_original",
    "avg_time_original",
    "time_reduction_sec",
    "time_reduction_percent"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 3) COMPUTE COMPUTATION TIME PERCENTAGE
# ============================================================

# Opposite of reduction percentage
df["computation_time_percent"] = 100 - df["time_reduction_percent"]

# Optional check using average times
df["computation_time_percent_check"] = (
    df["avg_time_predicted"] / df["avg_time_original"]
) * 100

# Difference should be very small, only rounding differences
df["difference_check"] = (
    df["computation_time_percent"] - df["computation_time_percent_check"]
).abs()

print("\nMaximum difference between 100 - reduction and direct calculation:")
print(df["difference_check"].max())

# ============================================================
# 4) GROUP BY nbES
# ============================================================

group_by_nbes = df.groupby("nbES", as_index=False).agg({
    "avg_time_predicted": "mean",
    "avg_time_original": "mean",
    "time_reduction_percent": "mean",
    "computation_time_percent": "mean"
})

# ============================================================
# 5) GROUP BY CONFIDENCE LEVEL
# ============================================================

group_by_confidence = df.groupby("confidence_level", as_index=False).agg({
    "avg_time_predicted": "mean",
    "avg_time_original": "mean",
    "time_reduction_percent": "mean",
    "computation_time_percent": "mean"
})

# ============================================================
# 6) GROUP BY p
# ============================================================

group_by_p = df.groupby("p", as_index=False).agg({
    "avg_time_predicted": "mean",
    "avg_time_original": "mean",
    "time_reduction_percent": "mean",
    "computation_time_percent": "mean"
})

# ============================================================
# 7) OPTIONAL: GROUP BY p AND nbES
# ============================================================

group_by_p_nbes = df.groupby(["p", "nbES"], as_index=False).agg({
    "avg_time_predicted": "mean",
    "avg_time_original": "mean",
    "time_reduction_percent": "mean",
    "computation_time_percent": "mean"
})

# ============================================================
# 8) OPTIONAL: GROUP BY p AND CONFIDENCE
# ============================================================

group_by_p_confidence = df.groupby(["p", "confidence_level"], as_index=False).agg({
    "avg_time_predicted": "mean",
    "avg_time_original": "mean",
    "time_reduction_percent": "mean",
    "computation_time_percent": "mean"
})

# ============================================================
# 9) PRINT RESULTS
# ============================================================

print("\nGrouped by nbES:")
print(group_by_nbes.to_string(index=False))

print("\nGrouped by confidence level:")
print(group_by_confidence.to_string(index=False))

print("\nGrouped by p:")
print(group_by_p.to_string(index=False))

print("\nGrouped by p and nbES:")
print(group_by_p_nbes.to_string(index=False))

print("\nGrouped by p and confidence level:")
print(group_by_p_confidence.to_string(index=False))

# ============================================================
# 10) SAVE RESULTS
# ============================================================

save_dir = r"D:\Savini\Wen Codes\savini_new_results"


group_by_p_nbes.to_csv(save_dir + r"\Computation_Time_Grouped_By_p_nbES_t.csv", index=False)
group_by_p_confidence.to_csv(save_dir + r"\Computation_Time_Grouped_By_p_Confidence_t.csv", index=False)

print("\nSaved CSV files successfully.")