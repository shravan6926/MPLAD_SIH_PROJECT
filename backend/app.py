from pathlib import Path
from datetime import datetime
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
FILES = {
    "recommended": PROJECT_ROOT / "mplads_recommended_works_2026-08-28.csv",
    "completed": PROJECT_ROOT / "mplads_completed_works_2026-08-28.csv",
    "expenditures": PROJECT_ROOT / "mplads_expenditures_2026-08-28.csv",
    "summary": PROJECT_ROOT / "mplads_mp_summary_2026-08-28.csv",
    "allocated": PROJECT_ROOT / "Allocated Limit for Honble MPs.csv",
}

def clean_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", value).strip()

def money(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce").fillna(0)

def column(frame, *parts):
    for name in frame.columns:
        lowered = name.lower()
        if all(part.lower() in lowered for part in parts):
            return name
    raise KeyError(f"Could not find column containing: {parts}")

def load_data():
    recommended = pd.read_csv(FILES["recommended"], low_memory=False)
    completed = pd.read_csv(FILES["completed"], low_memory=False)
    expenditures = pd.read_csv(FILES["expenditures"], low_memory=False)
    summary = pd.read_csv(FILES["summary"], low_memory=False)
    allocated = pd.read_csv(FILES["allocated"], low_memory=False)
    for frame in [recommended, completed, expenditures, summary, allocated]:
        frame.columns = [clean_text(c) for c in frame.columns]
    amount_columns = [
        (recommended, column(recommended, "recommended", "amount")),
        (completed, column(completed, "final", "amount")),
        (expenditures, column(expenditures, "expenditure", "amount")),
        (summary, column(summary, "total", "expenditure")),
        (summary, column(summary, "allocated", "amount")),
        (allocated, column(allocated, "allocated", "amount")),
    ]
    for frame, amount_column in amount_columns:
        frame[amount_column] = money(frame[amount_column])
    recommended_amount = column(recommended, "recommended", "amount")
    completed_amount = column(completed, "final", "amount")
    expenditure_amount = column(expenditures, "expenditure", "amount")
    allocated_amount = column(allocated, "allocated", "amount")
    recommended.rename(columns={recommended_amount: "Recommended Amount (INR)"}, inplace=True)
    completed.rename(columns={completed_amount: "Final Amount (INR)"}, inplace=True)
    expenditures.rename(columns={expenditure_amount: "Expenditure Amount (INR)"}, inplace=True)
    allocated.rename(columns={allocated_amount: "Allocated Amount (INR)"}, inplace=True)
    recommended["Recommendation Date"] = pd.to_datetime(recommended.get("Recommendation Date"), errors="coerce", utc=True)
    completed["Completed Date"] = pd.to_datetime(completed.get("Completed Date"), errors="coerce", utc=True)
    expenditures["Expenditure Date"] = pd.to_datetime(expenditures.get("Expenditure Date"), errors="coerce", utc=True)
    completed_by_id = completed.drop_duplicates("Work ID").set_index("Work ID")
    projects = recommended.copy()
    projects["Work ID"] = projects["Work ID"].astype(str)
    completed_by_id.index = completed_by_id.index.astype(str)
    projects["Final Amount"] = projects["Work ID"].map(completed_by_id["Final Amount (INR)"]).fillna(0)
    projects["Completed Date"] = projects["Work ID"].map(completed_by_id["Completed Date"])
    projects["completed"] = projects["Work ID"].isin(completed_by_id.index)
    projects["Expenditure"] = projects["Work ID"].map(expenditure_by_id(expenditures, projects, expenditure_amount)).fillna(0)
    projects["variance"] = np.where(projects["Recommended Amount (INR)"] > 0, (projects["Final Amount"] - projects["Recommended Amount (INR)"]) / projects["Recommended Amount (INR)"], 0)
    projects["duration"] = (projects["Completed Date"] - projects["Recommendation Date"]).dt.days
    projects["risk_score"], projects["risk_reasons"] = score_projects(projects, expenditures)
    projects["risk_level"] = pd.cut(projects["risk_score"], [-1, 30, 70, 101], labels=["Low", "Medium", "High"]).astype(str)
    return {"recommended": recommended, "completed": completed, "expenditures": expenditures, "summary": summary, "allocated": allocated, "projects": projects}

def expenditure_by_id(exp, projects, expenditure_amount):
    keys = ["MP Name", "Constituency", "State", "Work Description"]
    exp = exp.copy()
    for frame in [exp, projects]:
        frame["match_key"] = frame[keys].fillna("").astype(str).apply(lambda row: "|".join(clean_text(x).lower() for x in row), axis=1)
    return exp.groupby("match_key")["Expenditure Amount (INR)"].sum().rename(index=lambda x: x)

def score_projects(projects, expenditures):
    amount = projects["Recommended Amount (INR)"].replace(0, np.nan)
    financial = ((projects["Final Amount"] / amount - 1).clip(0, 1) * 45).fillna(0)
    missing_completion = (~projects["completed"]).astype(int) * 10
    long_duration = ((projects["duration"].fillna(0) - 365).clip(0, 730) / 730 * 20)
    features = projects[["Recommended Amount (INR)", "Final Amount", "Expenditure"]].fillna(0)
    if len(features) > 10:
        model = IsolationForest(n_estimators=80, contamination="auto", random_state=42, n_jobs=-1).fit(features)
        anomaly = pd.Series(-model.decision_function(features), index=projects.index).rank(pct=True) * 25
    else:
        anomaly = pd.Series(0, index=projects.index)
    scores = (financial + missing_completion + long_duration + anomaly).clip(0, 100).round().astype(int)
    reasons = []
    for idx, row in projects.iterrows():
        current = []
        if row["variance"] > 0.15: current.append(f"Final amount is {row['variance']:.0%} above recommendation")
        if row["Expenditure"] > row["Recommended Amount (INR)"] > 0: current.append("Matched expenditure exceeds the recommended amount")
        if pd.notna(row["duration"]) and row["duration"] > 365: current.append("Completion duration is above one year")
        if not row["completed"]: current.append("No matching completion record")
        if not current: current.append("No material signal detected in available fields")
        reasons.append(current)
    return scores, reasons

DATA = load_data()
app = FastAPI(title="MPLADS AI Monitor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def num(value):
    return round(float(value or 0), 2)

def project_json(row):
    return {"work_id": str(row["Work ID"]), "description": clean_text(row.get("Work Description")), "category": clean_text(row.get("Category")) or "Not available", "mp": clean_text(row.get("MP Name")), "constituency": clean_text(row.get("Constituency")), "state": clean_text(row.get("State")), "recommended": num(row.get("Recommended Amount (INR)")), "final": num(row.get("Final Amount")), "expenditure": num(row.get("Expenditure")), "recommendation_date": date_json(row.get("Recommendation Date")), "completed_date": date_json(row.get("Completed Date")), "status": "Completed" if row.get("completed") else "Recommended", "risk_score": int(row.get("risk_score", 0)), "risk_level": row.get("risk_level", "Low"), "reasons": row.get("risk_reasons", [])}

def date_json(value):
    return value.isoformat()[:10] if pd.notna(value) else None

def filtered_projects(state="all", risk="all", search=""):
    frame = DATA["projects"]
    if state != "all": frame = frame[frame["State"].astype(str) == state]
    if risk != "all": frame = frame[frame["risk_level"] == risk]
    if search:
        text = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        frame = frame[text.str.contains(search.lower(), regex=False)]
    return frame

@app.get("/")
def home(): return FileResponse(FRONTEND_ROOT / "index.html")

@app.get("/static/{asset_path:path}")
def frontend_asset(asset_path: str): return FileResponse(FRONTEND_ROOT / asset_path)

@app.get("/api/dashboard")
def dashboard(state: str = "all", risk: str = "all", search: str = ""):
    p = filtered_projects(state, risk, search)
    e = DATA["expenditures"]
    if state != "all": e = e[e["State"].astype(str) == state]
    allocated = DATA["allocated"]
    if state != "all": allocated = allocated[allocated["State"].astype(str) == state]
    total_allocated = allocated["Allocated Amount (INR)"].sum()
    expenditure = e["Expenditure Amount (INR)"].sum()
    by_state = p.groupby("State").agg(works=("Work ID", "count"), expenditure=("Expenditure", "sum"), completed=("completed", "sum")).reset_index().sort_values("expenditure", ascending=False).head(10)
    risk_dist = p["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    return {"kpis": {"recommended_works": len(p), "completed_works": int(p["completed"].sum()), "expenditure": num(expenditure), "allocated": num(total_allocated), "utilization": num(expenditure / total_allocated * 100 if total_allocated else 0), "high_risk": int((p["risk_level"] == "High").sum()), "payment_anomalies": int((e["Payment Status"].astype(str).str.contains("In-Progress|Failed", case=False, na=False)).sum())}, "risk_distribution": [{"name": k, "value": int(v)} for k, v in risk_dist.items()], "states": [{"state": clean_text(r["State"]), "works": int(r["works"]), "completed": int(r["completed"]), "expenditure": num(r["expenditure"])} for _, r in by_state.iterrows()], "categories": [{"name": clean_text(k), "value": int(v)} for k, v in p["Category"].fillna("Not available").value_counts().head(8).items()], "available_states": sorted(DATA["projects"]["State"].dropna().astype(str).unique().tolist())}

@app.get("/api/projects")
def projects(page: int = 1, limit: int = Query(25, le=100), state: str = "all", risk: str = "all", search: str = ""):
    p = filtered_projects(state, risk, search).sort_values("risk_score", ascending=False)
    start = (page - 1) * limit
    return {"total": len(p), "page": page, "pages": max(1, int(np.ceil(len(p) / limit))), "items": [project_json(r) for _, r in p.iloc[start:start + limit].iterrows()]}

@app.get("/api/projects/{work_id}")
def project(work_id: str):
    p = DATA["projects"][DATA["projects"]["Work ID"].astype(str) == work_id]
    if p.empty: return {"error": "Project not found"}
    result = project_json(p.iloc[0])
    result["entity_matching"] = "Expenditure matching uses description + MP + constituency + state because source expenditures have no Work ID."
    return result

@app.get("/api/alerts")
def alerts():
    p = DATA["projects"].sort_values("risk_score", ascending=False).head(30)
    return [{"id": f"ALT-{r['Work ID']}", "work_id": str(r["Work ID"]), "type": "Potential irregularity", "severity": r["risk_level"], "score": int(r["risk_score"]), "explanation": r["risk_reasons"][0], "status": "New", "action": "Review available financial and completion records"} for _, r in p[p["risk_score"] >= 50].iterrows()]

@app.get("/api/data-quality")
def quality():
    return {"recommended": len(DATA["recommended"]), "completed": len(DATA["completed"]), "expenditures": len(DATA["expenditures"]), "allocated": len(DATA["allocated"]), "missing_recommended_ids": int(DATA["recommended"]["Work ID"].isna().sum()), "missing_categories": int(DATA["recommended"]["Category"].isna().sum()), "invalid_recommendation_dates": int(DATA["recommended"]["Recommendation Date"].isna().sum()), "unmatched_expenditures": int((~DATA["expenditures"].index.isin([])).sum()), "coordinates": "Not available in supplied datasets"}

@app.get("/api/analytics/states")
def states(): return dashboard()["states"]

@app.get("/api/compliance")
def compliance():
    p = DATA["projects"]
    rows = []
    for _, r in p[p["variance"] > 0.15].sort_values("variance", ascending=False).head(100).iterrows():
        rows.append({"work_id": str(r["Work ID"]), "rule": "Potential cost variance", "severity": "Warning" if r["variance"] < .3 else "High", "explanation": f"Final amount is {r['variance']:.0%} above recommendation", "status": "Requires review"})
    return rows

@app.get("/api/health")
def health(): return {"status": "ok", "loaded_at": datetime.now().isoformat(), "records": {k: len(v) for k, v in DATA.items() if isinstance(v, pd.DataFrame)}}
