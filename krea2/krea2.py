import subprocess
import time
import atexit
import socket
import requests
from urllib.parse import urlparse
import paramiko, json, os
import threading
from pathlib import Path

CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN")
MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASS  = os.environ.get("MEGA_PASS")
NGROK_TOKEN = "3MHWInVH6i9DD0na4JuqvYNSJFC_3N7FKkZm1vnu6eCjJS7AA"
STOP_FLAG_PATH = "/root/STOP_SESSION"
KERNEL_JSON_LOCAL =r"ngrok\kaggle_engine\kaggle_kernel.json"

# このファイルと同じ階層の keys フォルダにあるログイン用の秘密鍵
BASE_DIR = Path(__file__).resolve().parent
SSH_KEY_PATH = str(BASE_DIR / "keys" / "kaggle_ngrok_key")

if not os.path.exists(SSH_KEY_PATH):
    raise FileNotFoundError(f"秘密鍵が見つかりません: {SSH_KEY_PATH}")

TOKEN_DATAS = {
    "CIVITAI": CIVITAI_TOKEN,
    "MEGA_EMAIL": MEGA_EMAIL,
    "MEGA_PASS": MEGA_PASS
}

def get_ngrok_URL():
    headers = {
        "Authorization": f"Bearer {NGROK_TOKEN}",
        "Ngrok-Version": "2",
    }

    url = None
    response = requests.get("https://api.ngrok.com/tunnels", headers=headers)
    response.raise_for_status()  # ステータスコードが200以外なら例外を出す
    data = response.json()

    if len(data["tunnels"]) > 0:
        url = data["tunnels"][0]["public_url"]
        return url
    else:
        return None
    
url = get_ngrok_URL()
if url is None:
    subprocess.run(r"python -m kaggle kernels push -p ngrok\kaggle_engine", text=True)

    attempt_count = 0
    while True:
        url = get_ngrok_URL()
        if url is not None:
            break
        else:
            if attempt_count > 120:
                print("トンネルが見つかりませんでした")
                exit()
            print("トンネルなし")
            attempt_count+=1

        time.sleep(1)

print(url)
parsed = urlparse(url)

SSH_HOST = parsed.hostname
SSH_PORT = parsed.port
SSH_USER = "root"
PORTS = [60001, 60002, 60003, 60004, 60005]

def connect_ssh(host, port, user="root", password=None, key_path=SSH_KEY_PATH):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host, port=port, username=user,
        key_filename=key_path,
        look_for_keys=False,   # ~/.ssh 内の他の鍵を探さない
        allow_agent=False,     # ssh-agent の鍵も使わない
        timeout=15,
    )
    return client

def download_kernel_json(host, port, local_path=KERNEL_JSON_LOCAL,
                          remote_path="/root/kaggle_kernel.json",
                          user="root", password=None, key_path=SSH_KEY_PATH):
    client = connect_ssh(host, port, user, password, key_path)
    try:
        sftp = client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
    finally:
        client.close()

    with open(local_path) as f:
        info = json.load(f)
    info["ip"] = "127.0.0.1"
    with open(local_path, "w") as f:
        json.dump(info, f)
    print(f"{local_path} にダウンロードしました")

def build_ssh_command():
    cmd = ["ssh", "-N", "-i", SSH_KEY_PATH, "-o", "IdentitiesOnly=yes"]
    for port in PORTS:
        cmd += ["-L", f"{port}:localhost:{port}"]
    cmd += [
        "-p", str(SSH_PORT),
        f"{SSH_USER}@{SSH_HOST}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "GlobalKnownHostsFile=NUL",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]
    return cmd

def wait_for_port(port, host="localhost", timeout=30):
    """指定ポートが開くまで待つ"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def start_tunnel():
    print("SSHトンネルを起動しています...")
    proc = subprocess.Popen(
        build_ssh_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # プロセスが即座に落ちていないかチェック
    time.sleep(1)
    if proc.poll() is not None:
        out, err = proc.communicate()
        raise RuntimeError(f"SSHトンネルの起動に失敗しました:\n{err.decode(errors='ignore')}")

    # 最初のポートが開くまで待機(接続確認)
    if not wait_for_port(PORTS[0]):
        proc.terminate()
        raise RuntimeError("トンネルは起動しましたが、ポートが開きませんでした。")

    print("SSHトンネルが接続されました。")
    return proc

def stop_tunnel(proc):
    if proc.poll() is None:
        print("SSHトンネルを終了します...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def stop_server(timeout=60):
    """Kaggle側に停止フラグを作成し、トンネルが消えるまで待つ"""
    print("Kaggleセッションの停止を指示します...")
    try:
        client = connect_ssh(SSH_HOST, SSH_PORT)
        try:
            _, stdout, _ = client.exec_command(f"touch {STOP_FLAG_PATH}")
            stdout.channel.recv_exit_status()
        finally:
            client.close()
    except Exception as e:
        print(f"停止フラグの作成に失敗しました: {e}")
        return False

    # Kaggle側は5秒間隔で検知 → sshd/ngrok停止。トンネルが消えるまで待つ
    start = time.time()
    while time.time() - start < timeout:
        try:
            if get_ngrok_URL() is None:
                print("サーバーが停止しました。")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("停止を確認できませんでした（Kaggle側で終了処理中の可能性があります）")
    return False

_cleanup_done = False
_cleanup_lock = threading.Lock()

def shutdown_all(tunnel_proc):
    """サーバー停止 → SSHトンネル終了 → ローカルの機密ファイル削除（1回だけ実行）"""
    global _cleanup_done
    with _cleanup_lock:
        if _cleanup_done:
            return
        _cleanup_done = True

    # 先にサーバーへ停止指示（トンネルとは別の直接SSH接続なので順序は自由）
    stop_server()
    stop_tunnel(tunnel_proc)

def subprocess_run(cmd):
    try:
        # 実行して完了を待つ (エラー時はCalledProcessErrorが発生)
        result = subprocess.run(cmd, check=True, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました (終了コード: {e.returncode})")
    except Exception as e:
        print(f"予期せぬエラー: {e}")

if __name__ == "__main__":
    tunnel_proc = start_tunnel()
    atexit.register(shutdown_all, tunnel_proc)  # 異常終了時の保険

    try:
        download_kernel_json(SSH_HOST, SSH_PORT)
        cmd = [
            "python", "-u",
            r"ngrok\execute_cell.py",
            r"main.py",
            "--data", json.dumps(TOKEN_DATAS, ensure_ascii=False),
            "--var", "tokens",
            "0","1","2","3","4","5"
        ]
        subprocess_run(cmd)

        cmd = [
            "python", "-u",
            r"ngrok\execute_cell.py",
            r"main.py", "6"
        ]
        while True:
            if input("準備完了(e=終了, other=生成)") != "e":
                subprocess_run(cmd)
            else:
                break
        print("終了します")

    finally:
        shutdown_all(tunnel_proc)