"""Basic exploratory analysis for the InfraGuardAI demo dataset."""
from pathlib import Path
import pandas as pd
from preprocess import load_dataset

def run_eda(csv_path=None):
    df=load_dataset(csv_path)
    print("INFRA GUARD AI - DEMO DATA EDA")
    print("Rows:",len(df))
    print("Maintenance required rate:",round(df.maintenance_required.mean()*100,2),"%")
    print("\nComplaints by asset type:")
    print(df.groupby("asset_type").complaint_count.agg(["sum","mean"]).sort_values("sum",ascending=False))
    print("\nAverage age by target:")
    print(df.groupby("maintenance_required").age_years.mean())
    print("\nCondition/failure correlation:",round(df.condition_score.corr(df.failure_count),3))
    print("\nMaintenance rate by type:")
    print((df.groupby("asset_type").maintenance_required.mean()*100).sort_values(ascending=False))

if __name__=="__main__": run_eda()
