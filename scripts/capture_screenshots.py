"""Capture report screenshots of the running stack.

Needs: docker compose stack up, k8s deployed, `mlflow ui --port 5005` running,
and `pip install playwright` (uses the locally installed Chrome).

Usage:  python scripts/capture_screenshots.py
"""
import html
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
TERM = OUT / "_terminal"

TERMINAL_CSS = """
body{margin:0;background:#1e1e2e;font:14px/1.45 Consolas,'Cascadia Mono',monospace;color:#cdd6f4}
.bar{background:#313244;padding:8px 14px;color:#a6adc8;font:13px 'Segoe UI',sans-serif}
.bar i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}
pre{margin:0;padding:14px 18px;white-space:pre-wrap;word-break:break-all}
.cmd{color:#89b4fa;font-weight:bold}.ok{color:#a6e3a1}.bad{color:#f38ba8}.hd{color:#f9e2af}
"""


def terminal_html(title, text):
    lines = []
    for raw in text.splitlines():
        line = html.escape(raw)
        if raw.startswith("$ "):
            line = f'<span class="cmd">{line}</span>'
        elif any(k in raw for k in ("FAILED", "Error", "exit=1", " F401", " E303", "failed")):
            line = f'<span class="bad">{line}</span>'
        elif raw.startswith("====="):
            line = f'<span class="hd">{line}</span>'
        elif any(k in raw for k in ("PASSED", "passed", " OK", "Running", "deployed", "healthy")):
            line = f'<span class="ok">{line}</span>'
        lines.append(line)
    dots = "".join(f'<i style="background:{c}"></i>' for c in ("#f38ba8", "#f9e2af", "#a6e3a1"))
    return (f"<html><head><style>{TERMINAL_CSS}</style></head><body>"
            f'<div class="bar">{dots} {html.escape(title)}</div>'
            f"<pre>{chr(10).join(lines)}</pre></body></html>")


def experiment_id():
    url = ("http://localhost:5005/api/2.0/mlflow/experiments/get-by-name"
           "?experiment_name=heart-disease")
    with urllib.request.urlopen(url) as r:
        return json.load(r)["experiment"]["experiment_id"]


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Terminal captures
        for txt in sorted(TERM.glob("*.txt")):
            page.set_viewport_size({"width": 1200, "height": 300})
            page.set_content(terminal_html(txt.stem[3:].replace("_", " "),
                                           txt.read_text(encoding="utf-8")))
            page.screenshot(path=OUT / f"{txt.stem}.png", full_page=True)
            print("terminal", txt.stem)
        page.set_viewport_size({"width": 1440, "height": 900})

        # Architecture diagram
        page.goto((ROOT / "report" / "architecture.html").as_uri())
        page.locator("#diagram").screenshot(
            path=ROOT / "report" / "figures" / "architecture.png")

        # MLflow
        exp = experiment_id()
        page.goto(f"http://localhost:5005/#/experiments/{exp}")
        page.wait_for_selector("text=logistic_regression", timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "01_mlflow_runs.png")

        page.get_by_text("logistic_regression").first.click()
        page.wait_for_selector("text=Metrics", timeout=30000)
        page.wait_for_timeout(2000)
        page.screenshot(path=OUT / "02_mlflow_run_detail.png", full_page=True)

        # Swagger: execute /predict
        page.goto("http://localhost:8010/docs")
        page.wait_for_selector("text=/predict")
        page.locator(".opblock-post").first.click()
        page.get_by_role("button", name="Try it out").click()
        body = (ROOT / "sample_request.json").read_text()
        page.locator("textarea.body-param__text").fill(body)
        page.get_by_role("button", name="Execute").click()
        page.wait_for_selector(".response .highlight-code", timeout=15000)
        page.wait_for_timeout(800)
        page.locator(".opblock-post").first.screenshot(path=OUT / "09_swagger_predict.png")

        # Prometheus
        page.goto("http://localhost:9095/targets")
        page.wait_for_timeout(2500)
        page.screenshot(path=OUT / "12_prometheus_targets.png")
        q = "sum by (endpoint, status) (rate(api_requests_total[1m]))"
        page.goto("http://localhost:9095/graph?g0.expr="
                  + urllib.request.quote(q) + "&g0.tab=0&g0.range_input=10m")
        page.wait_for_timeout(4000)
        page.screenshot(path=OUT / "13_prometheus_query.png")

        # Grafana
        page.goto("http://localhost:3005/d/heart-api?orgId=1&from=now-10m&to=now&kiosk")
        page.wait_for_timeout(8000)
        page.screenshot(path=OUT / "15_grafana_dashboard.png")

        browser.close()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
