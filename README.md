# CivicLense

A local decision-support dashboard built around the supplied MPLADS CSV datasets. It calculates live aggregates and explainable review signals; it does not label people or organizations as fraudulent.

## Run

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000. The API is documented at http://127.0.0.1:8000/docs.

`backend/` contains the FastAPI service and `frontend/` contains the browser UI. The importer normalizes the supplied fields at startup. Expenditure records have no Work ID in the source data, so matching is deliberately described as metadata-based and should be extended with a reviewable TF-IDF matching table for production. Coordinates are reported as unavailable because none are present in the supplied files.
