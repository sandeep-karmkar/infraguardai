"""Convert maintenance probability into InfraGuardAI risk."""
def calculate_risk(probability):
    p=max(0.0,min(1.0,float(probability)))
    score=int(round(p*100))
    if score<=30: category="LOW"; action="Continue routine monitoring and schedule preventive inspection."
    elif score<=60: category="MEDIUM"; action="Schedule preventive maintenance within 60-90 days."
    elif score<=80: category="HIGH"; action="Prioritize field inspection and maintenance within 14-30 days."
    else: category="CRITICAL"; action="Escalate for urgent field inspection and maintenance planning."
    return {"risk_score":score,"risk_category":category,"recommended_action":action}
