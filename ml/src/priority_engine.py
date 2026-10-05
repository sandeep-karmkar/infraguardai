"""InfraGuardAI priority engine: 50% risk + 30% impact + 20% urgency."""
ASSET_IMPACT={"Bridge":95,"Road":85,"Water system":80,"Drainage":70,"Public building":65,"Streetlight":45,"Other":50}
TIERS={"P1":{"label":"CRITICAL","sla_days":2},"P2":{"label":"HIGH","sla_days":10},"P3":{"label":"MEDIUM","sla_days":30},"P4":{"label":"LOW","sla_days":90}}

def calculate_priority(risk_score,asset_type,complaint_count=0,failure_count=0,usage_level="Medium",days_since_maintenance=0):
    risk=max(0,min(100,float(risk_score)))
    impact=float(ASSET_IMPACT.get(asset_type,50))
    if usage_level=="High": impact=min(100,impact*1.10)
    elif usage_level=="Low": impact*=.90
    complaint_signal=min(100,max(0,complaint_count)*5.0)
    failure_signal=min(100,max(0,failure_count)*20.0)
    maintenance_gap=min(100,max(0,days_since_maintenance)/365*100)
    urgency=complaint_signal*.45+failure_signal*.35+maintenance_gap*.20
    score=int(round(max(0,min(100,risk*.50+impact*.30+urgency*.20))))
    tier="P1" if score>=80 else "P2" if score>=60 else "P3" if score>=40 else "P4"
    meta=TIERS[tier]
    return {"priority_score":score,"priority_tier":tier,"priority_label":meta["label"],"sla_resolution_days":meta["sla_days"],"component_breakdown":{"risk_component":round(risk*.50,1),"impact_component":round(impact*.30,1),"urgency_component":round(urgency*.20,1)}}
