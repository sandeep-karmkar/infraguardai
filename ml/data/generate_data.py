"""Generate DEMO/SIMULATED infrastructure data for InfraGuardAI."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED=42
N_SAMPLES=3000
ASSET_TYPES=["Road","Bridge","Streetlight","Drainage","Water system","Public building","Other"]
USAGE_LEVELS=["Low","Medium","High"]
ENVIRONMENTAL_RISKS=["Low","Medium","High"]
BASE_LAT=23.3441
BASE_LON=85.3096

def generate_dataset(n_samples=N_SAMPLES, seed=SEED):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(1,n_samples+1):
        asset_type=rng.choice(ASSET_TYPES,p=[.30,.08,.20,.14,.12,.10,.06])
        age=int(rng.integers(1,31))
        usage=rng.choice(USAGE_LEVELS,p=[.25,.50,.25])
        env=rng.choice(ENVIRONMENTAL_RISKS,p=[.30,.45,.25])
        usage_stress={"Low":2,"Medium":8,"High":16}[usage]
        env_stress={"Low":0,"Medium":7,"High":15}[env]
        condition=int(np.clip(100-age*1.65-usage_stress-env_stress+rng.normal(0,7),10,100))
        days=int(np.clip(rng.gamma(3.0,55),5,650))
        maintenance_count=max(0,int(age/3+rng.normal(0,1.5)))
        complaint_lambda=max(.4,.5+(100-condition)*.10+(3 if usage=="High" else 0))
        complaints=int(rng.poisson(complaint_lambda))
        failure_lambda=max(.1,age/12+complaints/12+(.8 if condition<45 else 0))
        failures=int(rng.poisson(failure_lambda))
        inspection=int(np.clip(condition+rng.normal(0,5),10,100))
        latent=((100-condition)*.38+min(complaints*2.8,22)+min(failures*5,20)+min(days/365*18,18)+(100-inspection)*.12)
        probability=1/(1+np.exp(-((latent+rng.normal(0,6))-52)/9))
        target=int(rng.random()<probability)
        rows.append({"asset_id":f"AST-{i:05d}","asset_type":asset_type,"age_years":age,"condition_score":condition,"complaint_count":complaints,"failure_count":failures,"maintenance_count":maintenance_count,"days_since_maintenance":days,"inspection_score":inspection,"usage_level":usage,"environmental_risk":env,"latitude":round(BASE_LAT+rng.uniform(-.06,.06),6),"longitude":round(BASE_LON+rng.uniform(-.06,.06),6),"maintenance_required":target})
    return pd.DataFrame(rows)

if __name__=="__main__":
    out=Path(__file__).resolve().parent/'infrastructure_data.csv'
    df=generate_dataset()
    df.to_csv(out,index=False)
    print(f"Generated {len(df)} DEMO records -> {out}")
    print(f"Maintenance-required rate: {df.maintenance_required.mean()*100:.1f}%")
