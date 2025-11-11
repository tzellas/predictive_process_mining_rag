import pm4py
import pandas as pd


df = pd.read_csv("BPI_Challenge_2019.csv")
small_df = pd.DataFrame(columns=df.columns)
groups = []
for i, (case_id, group) in enumerate(df.groupby("case:concept:name")):
    groups.append(group)
    if i >= 62932:
        break

small_df = pd.concat(groups, ignore_index=True)
small_df.to_csv("Test_Data_BPI_Challenge_2019_.csv", index=False, encoding="utf-8")
    