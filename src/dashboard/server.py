"""
Dashboard: optional local web UI for ChurnLab.

Launches a lightweight HTML dashboard that displays:
  - Registered datasets
  - Experiment history
  - Leaderboards
  - Reports
"""
import http.server
import json
import os
import threading
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChurnLab Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f0f23; color: #e0e0e0; }
.header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
           padding: 24px 32px; border-bottom: 2px solid #0f3460; }
.header h1 { font-size: 24px; color: #00d2ff; }
.header p { color: #8892b0; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 32px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }
.card { background: #1a1a2e; border-radius: 12px; padding: 24px; border: 1px solid #233554; }
.card h2 { color: #00d2ff; font-size: 18px; margin-bottom: 16px; }
.card .stat { font-size: 36px; font-weight: bold; color: #ccd6f6; }
.card .label { font-size: 14px; color: #8892b0; margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #233554; }
th { color: #00d2ff; font-weight: 600; font-size: 13px; text-transform: uppercase; }
td { color: #ccd6f6; font-size: 14px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.badge-ok { background: #064e3b; color: #34d399; }
.badge-warn { background: #78350f; color: #fbbf24; }
.badge-err { background: #7f1d1d; color: #f87171; }
.section { margin-top: 32px; }
.section h2 { color: #ccd6f6; font-size: 20px; margin-bottom: 16px; border-bottom: 1px solid #233554; padding-bottom: 8px; }
footer { text-align: center; padding: 24px; color: #4a5568; font-size: 13px; border-top: 1px solid #233554; margin-top: 48px; }
</style>
</head>
<body>
<div class="header">
  <h1>ChurnLab Dashboard</h1>
  <p>Universal Customer Churn Research Framework</p>
</div>
<div class="container">
  <div class="grid">
    <div class="card">
      <h2>Registered Datasets</h2>
      <div class="stat" id="n-datasets">-</div>
      <div class="label">datasets available</div>
    </div>
    <div class="card">
      <h2>Experiments</h2>
      <div class="stat" id="n-experiments">-</div>
      <div class="label">experiments recorded</div>
    </div>
    <div class="card">
      <h2>Framework</h2>
      <div class="stat" id="version">-</div>
      <div class="label">current version</div>
    </div>
  </div>
  <div class="section">
    <h2>Datasets</h2>
    <table id="datasets-table">
      <thead><tr><th>Name</th><th>Ecosystem</th><th>Status</th></tr></thead>
      <tbody id="datasets-body"></tbody>
    </table>
  </div>
  <div class="section">
    <h2>Recent Experiments</h2>
    <table id="experiments-table">
      <thead><tr><th>ID</th><th>Dataset</th><th>Status</th><th>Duration</th></tr></thead>
      <tbody id="experiments-body"></tbody>
    </table>
  </div>
</div>
<footer>ChurnLab v1.0 &mdash; Universal Customer Churn Research Framework</footer>
<script>
async function loadData() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    document.getElementById('n-datasets').textContent = data.datasets.length;
    document.getElementById('n-experiments').textContent = data.experiments.length;
    document.getElementById('version').textContent = 'v' + data.version;
    const dsBody = document.getElementById('datasets-body');
    data.datasets.forEach(d => {
      dsBody.innerHTML += '<tr><td>' + d.name + '</td><td>' + d.ecosystem + '</td><td><span class="badge badge-ok">active</span></td></tr>';
    });
    const expBody = document.getElementById('experiments-body');
    data.experiments.slice(0, 10).forEach(e => {
      const badge = e.status === 'completed' ? 'badge-ok' : 'badge-err';
      expBody.innerHTML += '<tr><td>' + e.id + '</td><td>' + e.dataset + '</td><td><span class="badge ' + badge + '">' + e.status + '</span></td><td>' + e.duration + '</td></tr>';
    });
  } catch(e) { console.error('Failed to load data:', e); }
}
loadData();
</script>
</body></html>"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = self._collect_data()
            self.wfile.write(json.dumps(data, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _collect_data(self):
        from src.config import FRAMEWORK_VERSION
        datasets = []
        try:
            from src.datasets import list_datasets, get_dataset, get_ecosystem_type
            for name in list_datasets():
                eco = get_ecosystem_type(name)
                datasets.append({"name": name, "ecosystem": eco})
        except Exception:
            pass

        experiments = []
        exp_dir = ".experiments"
        if os.path.isdir(exp_dir):
            for fname in sorted(os.listdir(exp_dir), reverse=True)[:20]:
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(exp_dir, fname)) as f:
                            exp = json.load(f)
                        experiments.append({
                            "id": exp.get("experiment_id", fname),
                            "dataset": exp.get("dataset", "N/A"),
                            "status": exp.get("status", "unknown"),
                            "duration": f"{exp.get('runtime_seconds', 0):.0f}s",
                        })
                    except Exception:
                        pass

        return {
            "version": FRAMEWORK_VERSION,
            "datasets": datasets,
            "experiments": experiments,
        }

    def log_message(self, format, *args):
        pass


def launch_dashboard(port: int = 8420, open_browser: bool = True) -> None:
    """Launch the ChurnLab dashboard."""
    server = http.server.HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"ChurnLab Dashboard running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
