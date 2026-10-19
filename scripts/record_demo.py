"""Record a browser walkthrough of the pipeline (video/pipeline_demo.mp4).

Needs the same running services as capture_screenshots.py.
Usage:  python scripts/record_demo.py
"""
import shutil
import subprocess

from playwright.sync_api import sync_playwright

from capture_screenshots import ROOT, TERM, experiment_id, terminal_html

VIDEO_DIR = ROOT / "video"
STUDENT = "Jeyaram M · 2025AE05470"
COURSE = "Machine Learning Operations (MLOps) · AIMLCZG523 · Assignment 01"
REPO = "https://github.com/callmejei/mlops_2025ae05470_assignment"
SIZE = {"width": 1280, "height": 720}

CAPTION_JS = """([t, who]) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div'); el.id = '__cap';
    el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99999;display:flex;'
      + 'align-items:center;gap:16px;background:rgba(15,23,42,.9);color:#fff;'
      + 'font:600 19px Segoe UI,sans-serif;padding:12px 20px';
    el.innerHTML = '<span id="__t" style="flex:1;text-align:center"></span>'
      + '<span style="font:600 13px Segoe UI,sans-serif;color:#93c5fd;'
      + 'white-space:nowrap">'
      + who + '</span>';
    document.body.appendChild(el);
  }
  document.getElementById('__t').textContent = t;
}"""

TITLE = """<html style="background:#0f172a"><body style="margin:0;height:100vh;display:flex;
flex-direction:column;
align-items:center;justify-content:center;background:#0f172a;color:#fff;
font-family:Segoe UI,sans-serif;text-align:center">
<div style="font-size:18px;color:#93c5fd;font-weight:600;letter-spacing:1px">{course}</div>
<h1 style="font-size:44px;margin:14px 0 6px">{h}</h1>
<p style="font-size:21px;color:#94a3b8;margin:0 0 28px">{p}</p>
<div style="font-size:26px;font-weight:700">{student}</div>
<div style="font-size:16px;color:#94a3b8;margin-top:8px">{repo}</div></body></html>"""


def show_title(page, h, p, wait):
    # Load as a real document (not set_content on about:blank) so the first
    # recorded frames use the full viewport.
    card = VIDEO_DIR / "_raw" / "title.html"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(TITLE.format(course=COURSE, student=STUDENT, repo=REPO, h=h, p=p),
                    encoding="utf-8")
    page.goto(card.as_uri())
    page.wait_for_timeout(wait)


def caption(page, text, wait=4500):
    page.evaluate(CAPTION_JS, [text, STUDENT])
    page.wait_for_timeout(wait)


def show_terminal(page, name, text, scroll=False):
    page.set_content(terminal_html(name, (TERM / f"{name}.txt").read_text(encoding="utf-8")))
    caption(page, text, 3000)
    if scroll:
        for _ in range(6):
            page.mouse.wheel(0, 250)
            page.wait_for_timeout(700)
    page.wait_for_timeout(1500)


def main():
    tmp = VIDEO_DIR / "_raw"
    shutil.rmtree(tmp, ignore_errors=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(viewport=SIZE, record_video_dir=str(tmp),
                                  record_video_size=SIZE)
        page = ctx.new_page()

        show_title(page, "Heart Disease Risk — MLOps Pipeline",
                   "UCI data → EDA → MLflow → CI → Docker → Kubernetes → Monitoring", 5500)

        page.goto((ROOT / "report" / "architecture.html").as_uri())
        caption(page, "Architecture: data & experiments → CI/CD → serving & monitoring", 6000)

        show_terminal(page, "05_ci_pass",
                      "CI pipeline: lint → tests → data → EDA → train → docker build → smoke test",
                      scroll=True)
        show_terminal(page, "06_ci_fail",
                      "The pipeline fails fast on a lint error or failing test (exit code 1)")

        page.goto(REPO + "/actions/runs/39209666196")
        page.wait_for_timeout(3000)
        caption(page, "GitHub Actions: lint-test-train → docker, both green on every push", 4000)
        page.get_by_role("link", name="lint-test-train").first.click()
        page.wait_for_timeout(2500)
        caption(page, "Job steps: lint, unit tests, data, EDA, train & track, artifact upload", 4500)

        exp = experiment_id()
        page.goto(f"http://localhost:5005/#/experiments/{exp}")
        page.wait_for_selector("text=logistic_regression", timeout=30000)
        caption(page, "MLflow: every training run tracked (Logistic Regression vs Random Forest)")
        page.get_by_text("logistic_regression").first.click()
        page.wait_for_selector("text=Metrics", timeout=30000)
        caption(page, "Run detail: tuned params + CV and test metrics (ROC-AUC 0.911 CV / 0.961 test)", 3000)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(2500)

        show_terminal(page, "07_docker_compose",
                      "Docker Compose: API (:8010), Prometheus (:9095), Grafana (:3005)")

        page.goto("http://localhost:8010/docs")
        page.wait_for_selector("text=/predict")
        caption(page, "FastAPI service — Swagger UI", 2500)
        page.locator(".opblock-post").first.click()
        page.get_by_role("button", name="Try it out").click()
        page.locator("textarea.body-param__text").fill(
            (ROOT / "sample_request.json").read_text())
        caption(page, "POST /predict with a patient record (JSON)", 2500)
        page.get_by_role("button", name="Execute").click()
        page.wait_for_selector(".response .highlight-code", timeout=15000)
        page.locator(".live-responses-table").scroll_into_view_if_needed()
        caption(page, "Response: prediction = disease, confidence 0.95", 5000)

        show_terminal(page, "10_k8s_manifest",
                      "Kubernetes (docker-desktop): Deployment with 2 replicas + LoadBalancer on :8020")
        show_terminal(page, "11_k8s_helm",
                      "Same service from the Helm chart: helm/heart-api on :8030")
        show_terminal(page, "14_api_logs",
                      "Structured JSON logs for every request and prediction")

        page.goto("http://localhost:9095/targets")
        caption(page, "Prometheus scrapes /metrics every 5 s — target is UP")

        page.goto("http://localhost:3005/d/heart-api?orgId=1&from=now-15m&to=now&kiosk")
        page.wait_for_timeout(5000)
        caption(page, "Grafana: predictions, error rate, latency p50/p95, class mix", 6000)

        show_title(page, "Thank you",
                   "Code, notebooks, manifests and report in the repository", 4500)

        video = page.video.path()
        ctx.close()
        browser.close()

    out = VIDEO_DIR / "pipeline_demo.webm"
    shutil.move(video, out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out)
    to_mp4(out)


def to_mp4(webm):
    """Convert to H.264 MP4 if imageio-ffmpeg is installed (wider player support)."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("imageio-ffmpeg not installed; skipping MP4")
        return
    mp4 = webm.with_suffix(".mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)
    print("wrote", mp4)


if __name__ == "__main__":
    main()
