# Heart Disease Risk — End-to-End MLOps

**Machine Learning Operations (MLOps) · AIMLCZG523 · Assignment 01**
**Student:** Jeyaram M · **ID:** 2025AE05470

Repository: https://github.com/callmejei/mlops_2025ae05470_assignment

A classifier trained on the UCI Heart Disease (Cleveland) dataset. It's served as a FastAPI service in Docker, deployed to local Kubernetes and monitored with Prometheus and Grafana.

```
UCI CSV ──► src/data.py ──► src/eda.py ──► src/train.py ──► models/model.joblib
                                              │ (MLflow: params, metrics, plots, model)
                                              ▼
GitHub Actions: lint ► pytest ► train ► docker build ► smoke test
                                              ▼
              Docker image heart-disease-api ──► docker compose (API + Prometheus + Grafana)
                                             └─► Kubernetes (docker-desktop) Deployment + LoadBalancer
```

## Layout

| Path | Purpose |
|------|---------|
| `src/data.py` | Downloads the raw file, cleans it (missing `ca`/`thal` filled with the mode, target set to 1 when `num > 0`) |
| `src/eda.py` | Class balance, histograms, categorical counts and correlation heatmap, saved to `report/figures/` |
| `src/features.py` | `ColumnTransformer`: median impute + scale for numeric columns, mode impute + one-hot for categorical columns |
| `src/train.py` | GridSearchCV over Logistic Regression and Random Forest with 5-fold stratified CV, logged to MLflow |
| `src/predict.py` | Loads the saved pipeline and scores records |
| `api/main.py` | FastAPI app: `/predict`, `/health`, `/metrics`, with JSON request logs |
| `tests/` | Pytest tests for cleaning, preprocessing, training and the API |
| `notebooks/` | `01_eda`, `02_training`, `03_inference`, saved with their outputs |
| `k8s/deployment.yaml` | Namespace, Deployment (2 replicas, probes, resource limits) and LoadBalancer Service |
| `helm/heart-api/` | The same deployment as a Helm chart, with an optional Ingress |
| `scripts/ci_local.sh` | Runs the CI steps locally and stops on the first failure |
| `scripts/capture_screenshots.py`, `scripts/record_demo.py` | Regenerate `screenshots/` and `video/` from the running stack (Playwright + local Chrome) |
| `report/2025ae05470_report_local_pc.pdf` | Final written report |
| `monitoring/` | Prometheus scrape config and a pre-provisioned Grafana dashboard |
| `.github/workflows/ci.yml` | CI pipeline |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.data                # download + clean -> data/processed/heart_clean.csv
python -m src.eda                 # figures -> report/figures/
python -m src.train               # train, track, save models/model.joblib
pytest                            # unit tests
flake8 src api tests              # lint
mlflow ui --backend-store-uri ./mlruns --port 5005   # http://localhost:5005
```

## Results

The best model is chosen by cross-validated ROC-AUC, not by the small test set.

| Model | Best params | CV ROC-AUC | CV accuracy | Test ROC-AUC | Test accuracy | Test precision | Test recall |
|-------|-------------|-----------:|------------:|-------------:|--------------:|---------------:|------------:|
| **Logistic Regression** (selected) | C=0.1, L2 | **0.911** | 0.851 | 0.961 | 0.885 | 0.839 | 0.929 |
| Random Forest | 100 trees, max_depth=None, min_samples_leaf=3 | 0.900 | 0.818 | 0.945 | 0.869 | 0.813 | 0.929 |

Full numbers are in `models/metrics.json`.

## Run with Docker Compose

```bash
docker compose up -d --build
```

| Service | URL |
|---------|-----|
| API docs (Swagger) | http://localhost:8010/docs |
| Prometheus | http://localhost:9095 |
| Grafana (admin/admin) | http://localhost:3005/d/heart-api |

```bash
curl -X POST localhost:8010/predict -H "Content-Type: application/json" -d @sample_request.json
# {"prediction":1,"label":"disease","probability_disease":0.9478,"confidence":0.9478,"model_version":"1.0.0"}
```

## Deploy to local Kubernetes (Docker Desktop)

```bash
docker compose build api                                   # builds heart-disease-api:latest
kubectl --context docker-desktop apply -f k8s/deployment.yaml
kubectl --context docker-desktop -n heart-disease get pods,svc
curl -X POST localhost:8020/predict -H "Content-Type: application/json" -d @sample_request.json
```

Or install it with Helm:

```bash
helm --kube-context docker-desktop upgrade --install heart-api helm/heart-api -n heart-disease-helm --create-namespace
curl -X POST localhost:8030/predict -H "Content-Type: application/json" -d @sample_request.json
```

To remove them: `kubectl --context docker-desktop delete namespace heart-disease` and `helm --kube-context docker-desktop uninstall heart-api -n heart-disease-helm`.

## Monitoring

- Every request is logged as one JSON line with `request_id`, path, status and latency. Predictions also log the input and output (`docker compose logs api`).
- Prometheus metrics: `api_requests_total{method,endpoint,status}`, `api_request_latency_seconds`, `model_predictions_total{label}`.
- The Grafana dashboard shows total predictions, error rate, p50/p95 latency, the class split and request rate.

## API input

| Field | Meaning |
|-------|---------|
| age | years |
| sex | 1 = male, 0 = female |
| cp | chest pain type (1 typical angina … 4 asymptomatic) |
| trestbps | resting blood pressure (mmHg) |
| chol | serum cholesterol (mg/dl) |
| fbs | fasting blood sugar > 120 mg/dl |
| restecg | resting ECG (0–2) |
| thalach | max heart rate |
| exang | exercise-induced angina |
| oldpeak | ST depression |
| slope | slope of the peak ST segment (1–3) |
| ca | number of major vessels coloured (0–3) |
| thal | 3 normal, 6 fixed defect, 7 reversible defect |

Out-of-range or missing fields get a `422` response.

## Deliverables

| Item | Location |
|------|----------|
| Report (16 pages, PDF) | `report/2025ae05470_report_local_pc.pdf` (source `report/report.html`) |
| Lab implementation report (8 pages, PDF) | `report/2025ae05470_prayogshala_lab_report.pdf`: the pipeline reproduced on the BITS Prayogshala MLOps lab VM |
| Architecture diagram | `report/figures/architecture.png` |
| Screenshots | `screenshots/` (MLflow, tests, local and GitHub Actions CI runs, Docker, Swagger, Kubernetes, Helm, Prometheus, logs, Grafana) |
| Pipeline video | `video/pipeline_demo.mp4`, 1 min 48 s captioned walkthrough including GitHub Actions |
| CI logs | `ci_logs/` (passing run, and runs stopped by a lint error and by a failing test) |

## Run on the BITS Prayogshala lab VM

```bash
cd ~/workspace
git clone https://github.com/callmejei/mlops_2025ae05470_assignment.git
bash mlops_2025ae05470_assignment/scripts/lab_setup.sh 2>&1 | tee ~/workspace/lab_setup.log
```

This runs the CI pipeline, starts Docker Compose, deploys to minikube (manifest and Helm, exposed via `minikube tunnel`), starts the MLflow UI and checks every endpoint. The evidence is in `report/2025ae05470_prayogshala_lab_report.pdf` and `screenshots/lab/`.
