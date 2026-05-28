import pandas as pd

# predicted-data file
df = pd.read_csv("D:\Savini\Wen Codes\savini_new_results\Transformer_results\optimal schedule\Results.csv")

# average runtime for each p, nbES, confidence_level
avg_times = (
    df.groupby(["p", "nbES", "confidence_level"])["SolutionTime"]
      .agg(["count", "sum", "mean"])
      .reset_index()
      .rename(columns={
          "count": "num_scenarios",
          "sum": "total_time",
          "mean": "avg_time"
      })
)
# save to a separate CSV file
output_path = r"D:\Savini\Wen Codes\savini_new_results\Transformer_results\optimal schedule\Average_Runtime_Summary.csv"
avg_times.to_csv(output_path, index=False)

print("Saved to:", output_path)
print(avg_times)