import pandas as pd
import os

base = r"D:\Savini\Wen Codes\savini_new_results"
levels = [95, 90, 80, 70, 60]
nbes_values = [1, 3, 5, 7, 9]

# -----------------------------
# 1. Read original baseline summaries
# -----------------------------
baseline_all = []

for lv in levels:
    # change this filename to your actual summary file name inside each CC_xx folder
    summary_path = fr"{base}\CC_{lv}\ComputationTime_Summary.csv"
    
    # read the excel file
    summary_df = pd.read_csv(summary_path, header=None)
    
    # keep only first 5 rows and first 3 columns
    # col 0 = num_scenarios
    # col 1 = total_time
    # col 2 = original average runtime
    summary_df = summary_df.iloc[:5, :3].copy()
    summary_df.columns = ["num_scenarios_original", "total_time_original", "avg_time_original"]
    
    # assign nbES and confidence level
    summary_df["nbES"] = nbes_values
    summary_df["confidence_level"] = lv / 100.0
    
    baseline_all.append(summary_df)

baseline_df = pd.concat(baseline_all, ignore_index=True)

# make sure confidence values match format like 0.6, 0.7, ...
baseline_df["confidence_level"] = baseline_df["confidence_level"].round(2)

# -----------------------------
# 2. Read predicted/warm-start summary results
# -----------------------------
results_path = fr"{base}\Transformer_results\Optimal schedule\Average_Runtime_Summary.csv"
avg_times = pd.read_csv(results_path)

avg_times["confidence_level"] = avg_times["confidence_level"].astype(float).round(2)

avg_times = avg_times.rename(columns={
    "num_scenarios": "num_scenarios_predicted",
    "total_time": "total_time_predicted",
    "avg_time": "avg_time_predicted"
})

# -----------------------------
# 3. Merge both tables
# -----------------------------
comparison_df = avg_times.merge(
    baseline_df[["nbES", "confidence_level", "num_scenarios_original", "total_time_original", "avg_time_original"]],
    on=["nbES", "confidence_level"],
    how="left"
)

# -----------------------------
# 4. Compute reduction
# -----------------------------
comparison_df["time_reduction_sec"] = (
    comparison_df["avg_time_original"] - comparison_df["avg_time_predicted"]
)

comparison_df["time_reduction_percent"] = (
    (comparison_df["avg_time_original"] - comparison_df["avg_time_predicted"])
    / comparison_df["avg_time_original"] * 100
)

# optional rounding
comparison_df = comparison_df.round({
    "total_time_predicted": 2,
    "avg_time_predicted": 2,
    "total_time_original": 2,
    "avg_time_original": 2,
    "time_reduction_sec": 2,
    "time_reduction_percent": 2
})

# -----------------------------
# 5. Save to CSV
# -----------------------------
output_path = fr"{base}\Computation_Time_Comparison_t.csv"
comparison_df.to_csv(output_path, index=False)

print("Saved comparison file to:")
print(output_path)
print(comparison_df)


summary_by_p = (
    comparison_df.groupby("p")[[
        "avg_time_predicted",
        "avg_time_original"
    ]]
    .mean()
    .reset_index()
)

summary_by_p["time_reduction_sec"] = (
    summary_by_p["avg_time_original"] - summary_by_p["avg_time_predicted"]
)

summary_by_p["time_reduction_percent"] = (
    summary_by_p["time_reduction_sec"] / summary_by_p["avg_time_original"] * 100
)

summary_by_p = summary_by_p.round(2)

output_path_p = fr"{base}\Comparison_Overall_By_p.csv"
summary_by_p.to_csv(output_path_p, index=False)

print(summary_by_p)
print("Saved to:", output_path_p)