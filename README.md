# PathSnap

The existing `pipeline_updated.py` builds trip-local candidate features; training scripts and saved model artifacts are under `models/` and `artifacts/`. `backend.py` runs raw-trip inference with the saved XGBoost classifier and a graph-aware Viterbi decoder. `frontend.py` is the evaluator interface.

## Setup

On both laptops, install Python dependencies with `pip install -r requirements.txt`. For local development, the API automatically uses the included `Student_folder/Student folder/student_test` trips. Restart the API after changing configuration. For evaluation, on the API laptop place the reserved trip directories (each containing matching `.track`, `.nodes`, and `.arcs` files, with optional `.route`) under `evaluation_trips/`, or set `PATHSNAP_EVALUATION_DIR` to their parent directory. Keep this directory on the API laptop. The trained model defaults to `artifacts/xgboost_model.json`; override with `PATHSNAP_MODEL_PATH` if needed.

Start the API on laptop 1:

```powershell
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Start the UI on laptop 2:

```powershell
$env:PATHSNAP_API_URL = 'http://<API-LAPTOP-IP>:8000'
streamlit run frontend.py --server.address 0.0.0.0
```

The UI lists trips from `GET /trips` and requests prediction from `POST /trips/{trip_id}/predict`. Check `GET /health` for setup status. If both laptops are on different origins, set `PATHSNAP_CORS_ORIGINS` on laptop 1 to the UI origin. The UI can also enter the API URL in its sidebar.

The API computes candidate features directly from the raw trip, then applies XGBoost and Viterbi. A supplied `.route` file is used only after prediction to compute a weak alignment accuracy. It is never supplied to the feature builder or decoder. Missing candidates return null arc IDs and reduce coverage. GPS gaps above 30 seconds restart decoding. The green map line shows selected road segments in observation order; it does not insert unobserved connecting arcs.

Training comparison, tuning, and selection must be reviewed from recorded experiments before claiming a final result. The reserved evaluator data must never be used for training or model selection.