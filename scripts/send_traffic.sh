#!/usr/bin/env bash
# Send varied /predict requests (plus some invalid ones) so the monitoring
# dashboards have data.   Usage: bash scripts/send_traffic.sh [count] [url]
N=${1:-150}
URL=${2:-http://localhost:8010}
for i in $(seq 1 "$N"); do
  thal=7; [ $((i % 3)) = 0 ] && thal=3
  body=$(printf '{"age":%d,"sex":%d,"cp":%d,"trestbps":%d,"chol":%d,"fbs":%d,"restecg":%d,"thalach":%d,"exang":%d,"oldpeak":%d.%d,"slope":%d,"ca":%d,"thal":%d}' \
    $((35 + i % 40)) $((i % 2)) $((i % 4 + 1)) $((110 + i % 60)) $((180 + i % 150)) $((i % 2)) \
    $((i % 3)) $((190 - i % 80)) $((i % 2)) $((i % 4)) $((i % 10)) $((i % 3 + 1)) $((i % 4)) $thal)
  curl -s -o /dev/null -X POST "$URL/predict" -H 'Content-Type: application/json' -d "$body"
  if [ $((i % 10)) = 0 ]; then
    curl -s -o /dev/null -X POST "$URL/predict" -H 'Content-Type: application/json' -d '{"age":999}'
  fi
  sleep 0.5
done
echo "sent $N requests to $URL"
