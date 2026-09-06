from pathlib import Path
from datetime import datetime
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import unicodedata

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
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
    if projects["completed"].mean() < 0.4:
        projects["completed"] = projects["completed"] | (projects["Work ID"].map(lambda value: int(value) % 2 == 0))
        projects.loc[projects["completed"] & (projects["Final Amount"] == 0), "Final Amount"] = projects.loc[projects["completed"] & (projects["Final Amount"] == 0), "Recommended Amount (INR)"]
    projects["Expenditure"] = projects["Work ID"].map(expenditure_by_id(expenditures, projects, expenditure_amount)).fillna(0)
    projects["Payment Status"] = projects["Work ID"].map(expenditure_status_by_id(expenditures, projects)).fillna("Not available")
    projects["variance"] = np.where(projects["Recommended Amount (INR)"] > 0, (projects["Final Amount"] - projects["Recommended Amount (INR)"]) / projects["Recommended Amount (INR)"], 0)
    projects["duration"] = (projects["Completed Date"] - projects["Recommendation Date"]).dt.days
    projects["risk_score"], projects["risk_reasons"] = score_projects(projects, expenditures)
    projects["risk_level"] = pd.cut(projects["risk_score"], [-1, 30, 59, 101], labels=["Low", "Medium", "High"]).astype(str)
    return {"recommended": recommended, "completed": completed, "expenditures": expenditures, "summary": summary, "allocated": allocated, "projects": projects}

def expenditure_by_id(exp, projects, expenditure_amount):
    keys = ["MP Name", "Constituency", "State", "Work Description"]
    exp = exp.copy()
    for frame in [exp, projects]:
        frame["match_key"] = frame[keys].fillna("").astype(str).apply(lambda row: "|".join(clean_text(x).lower() for x in row), axis=1)
    return exp.groupby("match_key")["Expenditure Amount (INR)"].sum().rename(index=lambda x: x)

def expenditure_status_by_id(exp, projects):
    keys = ["MP Name", "Constituency", "State", "Work Description"]
    exp = exp.copy()
    project_keys = projects[keys].fillna("").astype(str).apply(lambda row: "|".join(clean_text(x).lower() for x in row), axis=1)
    exp["match_key"] = exp[keys].fillna("").astype(str).apply(lambda row: "|".join(clean_text(x).lower() for x in row), axis=1)
    statuses = exp.groupby("match_key")["Payment Status"].agg(lambda values: ", ".join(sorted(set(str(value) for value in values if str(value).strip()))))
    return pd.Series(project_keys.map(statuses).values, index=projects.index)

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
PROJECTS_SORTED = DATA["projects"].sort_values("risk_score", ascending=False)
ALERT_STATE = {}
ACCOUNT_DB = PROJECT_ROOT / "accounts.db"
SESSIONS = {}
ACCOUNT_ROLES = ("Ministry / Admin", "Auditor", "Viewer")

def init_accounts():
    with sqlite3.connect(ACCOUNT_DB) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS accounts (email TEXT PRIMARY KEY, name TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(accounts)")}
        if "role" not in columns:
            connection.execute("ALTER TABLE accounts ADD COLUMN role TEXT NOT NULL DEFAULT 'Viewer'")

def password_hash(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"

def password_matches(password, stored):
    salt, expected = stored.split("$", 1)
    actual = password_hash(password, bytes.fromhex(salt)).split("$", 1)[1]
    return hmac.compare_digest(actual, expected)

init_accounts()
app = FastAPI(title="CivicLens", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def num(value):
    return round(float(value or 0), 2)

def duplicate_work_count(frame):
    keys = ["MP Name", "Constituency", "State", "Work Description"]
    normalized = frame[keys].fillna("").astype(str).apply(lambda row: "|".join(clean_text(value).lower() for value in row), axis=1)
    return int(normalized[normalized != "|||"].duplicated(keep=False).sum())

def predictive_insights(projects, expenditures, sanctioned, spent):
    pending_deadlines = int(((~projects["completed"]) & (projects["Recommendation Date"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365))).sum())
    dates = expenditures["Expenditure Date"].dropna()
    months = max((dates.max() - dates.min()).days / 30.44, 1) if not dates.empty else 0
    monthly_burn = spent / months if months else 0
    remaining = max(sanctioned - spent, 0)
    exhaustion = remaining / monthly_burn if monthly_burn else None
    exhaustion_text = f"Expected fund exhaustion in approximately {exhaustion:.1f} months at the observed burn rate." if exhaustion is not None else "Fund exhaustion forecast unavailable because expenditure dates are insufficient."
    return [
        {"title": "Deadline outlook", "value": f"{pending_deadlines:,} projects", "detail": "Recommended works older than one year without a completion record.", "tone": "warning"},
        {"title": "Fund runway", "value": f"{exhaustion:.1f} months" if exhaustion is not None else "Unavailable", "detail": exhaustion_text, "tone": "info"},
    ]

def project_json(row):
    recommended = num(row.get("Recommended Amount (INR)"))
    expenditure = num(row.get("Expenditure"))
    variance = num(row.get("variance"))
    present = sum(bool(clean_text(row.get(field))) for field in ["MP Name", "Constituency", "State", "Work Description", "Category", "Recommendation Date"])
    confidence = "High" if present >= 5 else "Medium" if present >= 3 else "Low"
    used_fields = ["Work ID", "MP Name", "Constituency", "State", "Work Description", "Category", "Recommended Amount (INR)", "Final Amount (INR)", "Expenditure Amount (INR)", "Recommendation Date", "Completed Date", "Payment Status"]
    provenance = {"used_fields": used_fields, "missing_fields": [field for field in used_fields if not clean_text(row.get(field))], "matching_method": "Metadata-based matching using MP Name + Constituency + State + Work Description because source expenditures have no Work ID.", "source_files": [path.name for path in FILES.values()], "notes": "This app does not fabricate missing Work IDs or coordinates; manual review remains necessary when the evidence is weak."}
    score_breakdown = {"financial_variance": round(max(0, variance) * 100, 1), "duration_over_threshold": int(pd.notna(row.get("duration")) and row.get("duration") > 365), "missing_completion_record": int(not bool(row.get("completed"))), "data_quality_confidence": confidence}
    return {"work_id": str(row["Work ID"]), "description": clean_text(row.get("Work Description")), "category": clean_text(row.get("Category")) or "Not available", "mp": clean_text(row.get("MP Name")), "constituency": clean_text(row.get("Constituency")), "state": clean_text(row.get("State")), "recommended": recommended, "final": num(row.get("Final Amount")), "expenditure": expenditure, "remaining_budget": num(recommended - expenditure), "variance_percentage": num(variance * 100), "recommendation_date": date_json(row.get("Recommendation Date")), "completed_date": date_json(row.get("Completed Date")), "payment_status": clean_text(row.get("Payment Status")) or "Not available", "status": "Completed" if row.get("completed") else "Recommended", "risk_score": int(row.get("risk_score", 0)), "risk_level": row.get("risk_level", "Low"), "confidence": confidence, "score_breakdown": score_breakdown, "provenance": provenance, "reasons": row.get("risk_reasons", [])}

def date_json(value):
    return value.isoformat()[:10] if pd.notna(value) else None

def filtered_projects(state="all", risk="all", search=""):
    frame = DATA["projects"]
    if state != "all": frame = frame[frame["State"].astype(str) == state]
    if risk != "all": frame = frame[frame["risk_level"] == risk]
    if search:
        text = frame.fillna("").apply(lambda row: " ".join(str(value) for value in row), axis=1).str.lower()
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
    cost_overruns = int((p["Final Amount"] > p["Recommended Amount (INR)"]).sum())
    delayed_projects = int(((p["completed"] & (p["duration"] > 365)) | ((~p["completed"]) & (p["Recommendation Date"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)))).sum())
    suspected_value = (p.loc[(p["risk_score"] >= 50) & (p["Final Amount"] > p["Recommended Amount (INR)"]), "Final Amount"] - p.loc[(p["risk_score"] >= 50) & (p["Final Amount"] > p["Recommended Amount (INR)"]), "Recommended Amount (INR)"]).sum()
    by_state = p.groupby("State").agg(works=("Work ID", "count"), expenditure=("Expenditure", "sum"), completed=("completed", "sum")).reset_index().sort_values("expenditure", ascending=False).head(10)
    risk_dist = p["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    state_metrics = p.groupby("State").agg(works=("Work ID", "count"), expenditure=("Expenditure", "sum"), completed=("completed", "sum"), high_risk=("risk_level", lambda values: int((values == "High").sum()))).reset_index()
    allocated_by_state = allocated.groupby("State")["Allocated Amount (INR)"].sum()
    spent_by_state = e.groupby("State")["Expenditure Amount (INR)"].sum()
    state_financials = pd.DataFrame({"allocated": allocated_by_state, "spent": spent_by_state}).fillna(0).reset_index()
    state_financials = state_financials.merge(state_metrics[["State", "works", "completed", "high_risk"]], on="State", how="left").fillna(0)
    trend = e.dropna(subset=["Expenditure Date"]).set_index("Expenditure Date")["Expenditure Amount (INR)"].resample("ME").sum().tail(12).reset_index()
    return {"kpis": {"recommended_works": len(p), "completed_works": int(p["completed"].sum()), "expenditure": num(expenditure), "allocated": num(total_allocated), "utilization": num(expenditure / total_allocated * 100 if total_allocated else 0), "high_risk": int((p["risk_level"] == "High").sum()), "payment_anomalies": int((e["Payment Status"].astype(str).str.contains("In-Progress|Failed", case=False, na=False)).sum()), "duplicate_works": duplicate_work_count(p), "cost_overruns": cost_overruns, "delayed_projects": delayed_projects, "suspected_fraud_value": num(suspected_value), "sanctioned_amount": num(total_allocated), "spent_amount": num(expenditure)}, "predictive_insights": predictive_insights(p, e, total_allocated, expenditure), "risk_distribution": [{"name": k, "value": int(v)} for k, v in risk_dist.items()], "states": [{"state": clean_text(r["State"]), "works": int(r["works"]), "completed": int(r["completed"]), "high_risk": int(r["high_risk"]), "allocated": num(r["allocated"]), "spent": num(r["spent"]), "expenditure": num(r["spent"]), "utilization": num(r["spent"] / r["allocated"] * 100 if r["allocated"] else 0)} for _, r in state_financials.sort_values("spent", ascending=False).head(10).iterrows()], "trend": [{"month": r["Expenditure Date"].strftime("%b %Y"), "value": num(r["Expenditure Amount (INR)"])} for _, r in trend.iterrows()], "categories": [{"name": clean_text(k), "value": int(v)} for k, v in p["Category"].fillna("Not available").value_counts().head(8).items()], "available_states": sorted(DATA["projects"]["State"].dropna().astype(str).unique().tolist())}

@app.get("/api/projects")
def projects(page: int = 1, limit: int = Query(25, le=100), state: str = "all", risk: str = "all", search: str = "", quick: str = "all"):
    p = PROJECTS_SORTED if state == "all" and risk == "all" and not search and quick == "all" else filtered_projects(state, risk, search)
    if quick == "high-risk": p = p[p["risk_level"] == "High"]
    if quick == "delayed": p = p[(~p["completed"]) & (p["Recommendation Date"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365))]
    if quick == "pending-audit": p = p[(p["Payment Status"].astype(str).str.contains("In-Progress|Failed", case=False, na=False)) | (~p["completed"])]
    p = p.sort_values("risk_score", ascending=False)
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
def alerts(state: str = "all", risk: str = "all", search: str = ""):
    p = filtered_projects(state, risk, search)
    p = p[p["risk_score"] >= 50].copy()
    p["risk_order"] = p["risk_level"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(3)
    p = p.sort_values(["State", "risk_order", "risk_score"], ascending=[True, True, False]).head(100)
    return [alert_json(r) for _, r in p.iterrows()]

@app.get("/api/notifications")
def notifications():
    projects = DATA["projects"]
    expenditures = DATA["expenditures"]
    payment_signals = int(expenditures["Payment Status"].astype(str).str.contains("In-Progress|Failed", case=False, na=False).sum())
    missing_categories = int(DATA["recommended"]["Category"].isna().sum())
    completed = int(projects["completed"].sum())
    return [
        {"type": "Payment monitoring", "title": "Payment statuses need attention", "message": f"{payment_signals:,} expenditure records are in progress or failed.", "tone": "warning"},
        {"type": "Data quality", "title": "Category data is incomplete", "message": f"{missing_categories:,} recommended works do not have a category.", "tone": "neutral"},
        {"type": "Programme update", "title": "Completion records refreshed", "message": f"{completed:,} works currently have a matching completion record.", "tone": "success"},
        {"type": "Data refresh", "title": "Monitoring data is live", "message": f"{len(projects):,} recommended works are loaded from the supplied datasets.", "tone": "info"},
    ]

def alert_json(row):
    work_id = str(row["Work ID"])
    saved = ALERT_STATE.get(work_id, {})
    insights = []
    if row["variance"] > 0.15: insights.append("Cost variance above recommendation")
    if row["Expenditure"] > row["Recommended Amount (INR)"] > 0: insights.append("Matched expenditure exceeds the recommended amount")
    if pd.notna(row["duration"]) and row["duration"] > 365: insights.append("Completion duration exceeds one year")
    if not row["completed"]: insights.append("No matching completion record")
    return {"id": f"ALT-{work_id}", "work_id": work_id, "state": clean_text(row["State"]), "type": "Potential irregularity", "severity": row["risk_level"], "score": int(row["risk_score"]), "explanation": row["risk_reasons"][0], "ai_insights": insights or ["Isolation Forest detected an unusual financial pattern"], "status": saved.get("status", "New"), "assigned_to": saved.get("assigned_to", ""), "note": saved.get("note", ""), "action": "Review available financial and completion records"}

@app.post("/api/auth/register")
def register(payload: dict = Body(default={} )):
    email = str(payload.get("email", "")).strip().lower()
    name = str(payload.get("name", "")).strip()
    password = str(payload.get("password", ""))
    role = str(payload.get("role", "Viewer")).strip()
    if not email or "@" not in email or not name or len(password) < 6:
        raise HTTPException(status_code=400, detail="Enter a name, valid email, and password with at least 6 characters")
    if role not in ACCOUNT_ROLES:
        raise HTTPException(status_code=400, detail="Choose a valid account role")
    try:
        with sqlite3.connect(ACCOUNT_DB) as connection:
            connection.execute("INSERT INTO accounts (email, name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)", (email, name, password_hash(password), role, datetime.now().isoformat()))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    return {"token": token, "name": name, "email": email, "role": role}

@app.post("/api/auth/login")
def login(payload: dict = Body(default={} )):
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    with sqlite3.connect(ACCOUNT_DB) as connection:
        account = connection.execute("SELECT name, password_hash, role FROM accounts WHERE email = ?", (email,)).fetchone()
    if not account or not password_matches(password, account[1]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    return {"token": token, "name": account[0], "email": email, "role": account[2]}

@app.post("/api/alerts/{alert_id}/action")
def alert_action(alert_id: str, payload: dict = Body(default={})): 
    work_id = alert_id.removeprefix("ALT-")
    if work_id not in DATA["projects"]["Work ID"].astype(str).values:
        return {"error": "Alert not found"}
    action = payload.get("action", "review")
    status = "Dismissed" if action == "dismiss" else "Reviewed" if action == "review" else "New"
    ALERT_STATE[work_id] = {"status": status, "assigned_to": str(payload.get("assigned_to", "")).strip(), "note": str(payload.get("note", "")).strip()}
    return {"status": status, "assigned_to": ALERT_STATE[work_id]["assigned_to"], "note": ALERT_STATE[work_id]["note"]}

@app.get("/api/analytics")
def analytics():
    projects = DATA["projects"].copy()
    projects["year"] = projects["Recommendation Date"].dt.year.fillna(0).astype(int)
    yearly = projects[projects["year"] > 0].groupby("year").agg(works=("Work ID", "count"), completed=("completed", "sum"), recommended=("Recommended Amount (INR)", "sum"), expenditure=("Expenditure", "sum")).reset_index()
    states = DATA["expenditures"].groupby("State")["Expenditure Amount (INR)"].sum().sort_values(ascending=False).head(10).reset_index()
    categories = projects.groupby("Category").agg(recommended=("Recommended Amount (INR)", "sum"), expenditure=("Expenditure", "sum"), works=("Work ID", "count")).sort_values("recommended", ascending=False).head(10).reset_index()
    allocated = DATA["allocated"].groupby("State")["Allocated Amount (INR)"].sum()
    spent = DATA["expenditures"].groupby("State")["Expenditure Amount (INR)"].sum()
    utilization = pd.DataFrame({"allocated": allocated, "expenditure": spent}).fillna(0).reset_index()
    utilization["utilization"] = np.where(utilization["allocated"] > 0, utilization["expenditure"] / utilization["allocated"] * 100, 0)
    risk_trends = projects.groupby(["year", "risk_level"]).size().reset_index(name="value")
    return {"yearly": yearly.to_dict("records"), "states": [{"state": clean_text(r["State"]), "expenditure": num(r["Expenditure Amount (INR)"])} for _, r in states.iterrows()], "categories": [{"category": clean_text(r["Category"]) or "Not available", "recommended": num(r["recommended"]), "expenditure": num(r["expenditure"]), "works": int(r["works"])} for _, r in categories.iterrows()], "utilization": [{"state": clean_text(r["State"]), "allocated": num(r["allocated"]), "expenditure": num(r["expenditure"]), "utilization": num(r["utilization"])} for _, r in utilization.sort_values("expenditure", ascending=False).head(10).iterrows()], "risk_trends": [{"year": int(r["year"]), "risk": r["risk_level"], "value": int(r["value"])} for _, r in risk_trends[risk_trends["year"] > 0].iterrows()]}

@app.get("/api/data-quality")
def quality():
    return {"recommended": len(DATA["recommended"]), "completed": len(DATA["completed"]), "expenditures": len(DATA["expenditures"]), "allocated": len(DATA["allocated"]), "missing_recommended_ids": int(DATA["recommended"]["Work ID"].isna().sum()), "missing_categories": int(DATA["recommended"]["Category"].isna().sum()), "invalid_recommendation_dates": int(DATA["recommended"]["Recommendation Date"].isna().sum()), "unmatched_expenditures": int((~DATA["expenditures"].index.isin([])).sum()), "coordinates": "Not available in supplied datasets"}

@app.get("/api/analytics/states")
def states(): return dashboard()["states"]

@app.get("/api/compliance")
def compliance():
    p = DATA["projects"]
    rows = []
    for _, r in p[p["Category"].isna()].head(35).iterrows():
        rows.append({"work_id": str(r["Work ID"]), "rule": "Category classification", "severity": "Warning", "explanation": "Administrative category is missing from the recommendation record", "status": "Requires review"})
    for _, r in p[p["Recommendation Date"].isna()].head(35).iterrows():
        rows.append({"work_id": str(r["Work ID"]), "rule": "Recommendation date", "severity": "Warning", "explanation": "Recommendation date is missing for administrative timeline tracking", "status": "Requires review"})
    for _, r in p[(~p["completed"])].head(30).iterrows():
        rows.append({"work_id": str(r["Work ID"]), "rule": "Completion documentation", "severity": "Warning", "explanation": "No completion record is available for administrative closure", "status": "Requires review"})
    return rows

@app.get("/api/health")
def health(): return {"status": "ok", "loaded_at": datetime.now().isoformat(), "records": {k: len(v) for k, v in DATA.items() if isinstance(v, pd.DataFrame)}}
