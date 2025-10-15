import os, subprocess, time, requests, sys, signal

def test_smoke_run_server_and_ask():
    # Build index first
    subprocess.check_call([sys.executable, "app/index/build_index.py", "--src", "data", "--out", "app/index/store"])

    # Start server
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8010"])
    try:
        # wait a moment
        time.sleep(2.5)
        r = requests.get("http://127.0.0.1:8010/health", timeout=10)
        assert r.status_code == 200 and r.json().get("ok") is True

        r = requests.post("http://127.0.0.1:8010/ask", json={"query": "How to deploy?", "top_k": 2}, timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data and "sources" in data
        assert isinstance(data["sources"], list)
    finally:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
