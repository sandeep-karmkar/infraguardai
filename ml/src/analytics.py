"""Analytics and GIS payloads for the InfraGuardAI dashboard."""
from pathlib import Path
import pandas as pd
from predict import predict_asset
from preprocess import load_dataset

def compute_batch_predictions(df):
    return [predict_asset(row) for row in df.to_dict(orient="records")]

def get_dashboard_analytics(csv_path=None):
    df=load_dataset(csv_path); scored=pd.DataFrame(compute_batch_predictions(df))
    risk={c:int((scored.risk_category==c).sum()) for c in ["CRITICAL","HIGH","MEDIUM","LOW"]}
    priority={p:int((scored.priority_tier==p).sum()) for p in ["P1","P2","P3","P4"]}
    breakdown={}
    for asset_type,group in df.groupby("asset_type"):
        ids=set(group.asset_id); s=scored[scored.asset_id.isin(ids)]
        breakdown[asset_type]={"total_count":len(group),"avg_condition":round(float(group.condition_score.mean()),1),"critical_count":int((s.risk_category=="CRITICAL").sum()),"total_complaints":int(group.complaint_count.sum())}
    return {"kpis":{"total_assets":len(df),"critical_risk_assets":risk["CRITICAL"],"high_risk_assets":risk["HIGH"],"urgent_maintenance_work_orders":priority["P1"]+priority["P2"],"avg_infrastructure_condition":round(float(df.condition_score.mean()),1),"total_citizen_complaints":int(df.complaint_count.sum())},"risk_distribution":risk,"priority_distribution":priority,"asset_breakdown":breakdown}

def get_gis_map_points(csv_path=None):
    df=load_dataset(csv_path); scored=compute_batch_predictions(df); points=[]
    for row,p in zip(df.to_dict(orient="records"),scored):
        points.append({"asset_id":row["asset_id"],"asset_type":row["asset_type"],"latitude":float(row["latitude"]),"longitude":float(row["longitude"]),"risk_score":p["risk_score"],"risk_category":p["risk_category"],"priority_tier":p["priority_tier"],"priority_label":p["priority_label"],"sla_days":p["sla_resolution_days"],"condition_score":int(row["condition_score"]),"complaint_count":int(row["complaint_count"]),"reasons":p["reasons"][:2]})
    return points
