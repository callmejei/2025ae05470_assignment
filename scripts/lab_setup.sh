#!/usr/bin/env bash
# Reproduce the whole pipeline on the BITS Prayogshala MLOps lab VM
# (Rocky Linux 9, conda Python 3.12, Docker, minikube).
#
#   cd ~/workspace && git clone https://github.com/callmejei/2025ae05470_assignment.git
#   bash ~/workspace/2025ae05470_assignment/scripts/lab_setup.sh
#
# The repo lives in ~/workspace (persistent). The virtualenv and helm binary
# live outside it: they are fast to recreate and EFS is slow for many small files.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
VENV=${VENV:-$HOME/.venvs/heart}
BIN=$HOME/.local/bin
export PATH=$BIN:$PATH
step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "Python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
fi
"$VENV/bin/pip" install -q -r requirements.txt
"$VENV/bin/python" --version

step "CI pipeline: lint, tests, data, EDA, train, docker build, smoke test"
PY="$VENV/bin/python" bash scripts/ci_local.sh

step "Docker Compose: API :8010, Prometheus :9095, Grafana :3005"
docker compose up -d --build
for i in $(seq 1 30); do curl -sf localhost:8010/health >/dev/null && break; sleep 2; done
docker compose ps

step "Kubernetes: minikube"
minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=4096
minikube image load heart-disease-api:latest
mkdir -p "$BIN"
cat > "$BIN/kubectl" <<'EOF'
#!/usr/bin/env bash
exec minikube kubectl -- "$@"
EOF
chmod +x "$BIN/kubectl"
kubectl apply -f k8s/deployment.yaml
kubectl -n heart-disease rollout status deploy/heart-api --timeout=180s

step "LoadBalancer tunnel (minikube tunnel, background)"
pgrep -f "minikube tunnel" >/dev/null || nohup minikube tunnel >"$HOME/minikube-tunnel.log" 2>&1 &
for i in $(seq 1 30); do
  ip=$(kubectl -n heart-disease get svc heart-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  [ -n "$ip" ] && break; sleep 2
done

step "Helm chart"
if ! command -v helm >/dev/null; then
  curl -sSL https://get.helm.sh/helm-v3.16.2-linux-amd64.tar.gz | tar -xz -C /tmp
  mv /tmp/linux-amd64/helm "$BIN/helm"
fi
helm upgrade --install heart-api helm/heart-api -n heart-disease-helm --create-namespace \
  --wait --timeout 180s
kubectl get svc -A | grep heart-api

step "MLflow UI :5005 (background)"
pgrep -f "mlflow ui" >/dev/null || nohup "$VENV/bin/mlflow" ui --backend-store-uri "$REPO/mlruns" \
  --host 0.0.0.0 --port 5005 >"$HOME/mlflow-ui.log" 2>&1 &

step "Verify endpoints"
LB1=$(kubectl -n heart-disease get svc heart-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
LB2=$(kubectl -n heart-disease-helm get svc heart-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
for url in localhost:8010 "$LB1:8020" "$LB2:8030"; do
  printf '%-22s ' "$url"
  curl -sf -m 10 -X POST "http://$url/predict" -H 'Content-Type: application/json' \
    -d @sample_request.json || echo "unreachable"
  echo
done

step "DONE"
cat <<EOF
API docs      http://localhost:8010/docs
Prometheus    http://localhost:9095
Grafana       http://localhost:3005/d/heart-api   (admin/admin)
MLflow        http://localhost:5005
K8s manifest  http://$LB1:8020/predict
K8s Helm      http://$LB2:8030/predict
EOF
