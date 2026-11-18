import json, os
import sys
from jupyter_client import BlockingKernelClient
import subprocess
CONN_FILE = r"kaggle_engine\kaggle_kernel.json"

def split_cells(path):
    """`# %%` で区切ったセルのリスト [(タイトル, コード), ...] を返す"""
    cells, title, cur = [], "", []
    started = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# %%"):
                if started and "".join(cur).strip():
                    cells.append((title, "".join(cur)))
                title = line[4:].strip()
                cur = []
                started = True
            else:
                cur.append(line)
    if "".join(cur).strip():
        cells.append((title, "".join(cur)))
    out = []
    for t, c in cells:
        first = c.strip().splitlines()[0][:40]
        out.append((t or first, c))
    return out


def connect():
    kc = BlockingKernelClient(connection_file=CONN_FILE)
    kc.load_connection_file()
    kc.start_channels()
    kc.hb_channel.time_to_dead = 15.0
    kc.wait_for_ready(timeout=60)
    return kc


def push_data(kc, data: dict, var_name: str = "input_data"):
    """辞書データをJSON経由でカーネルの名前空間に変数として注入する"""
    payload = json.dumps(data, ensure_ascii=False)
    code = (
        "import json as _json_for_push_data\n"
        f"{var_name} = _json_for_push_data.loads({payload!r})\n"
        "del _json_for_push_data"
    )
    run_cell(kc, code)


def run_cell(kc, code: str):
    msg_id = kc.execute(code)
    while True:
        msg = kc.get_iopub_msg(timeout=6000)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        msg_type = msg["msg_type"]
        if msg_type == "stream":
            print(msg["content"]["text"], end="")
        elif msg_type in ("execute_result", "display_data"):
            print(msg["content"]["data"].get("text/plain", ""))
        elif msg_type == "error":
            print("\n".join(msg["content"]["traceback"]))
        elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
            break


def run_file(kc, path, indexes=None, data: dict = None, var_name: str = "input_data"):
    """指定ファイルを # %% で分割して実行。indexesを指定すればそのセルだけ実行
    dataを指定すると、セル実行前にvar_name変数としてカーネルに注入する"""
    if data is not None:
        print(f"\n--- 辞書データを変数 '{var_name}' として注入 ---")
        push_data(kc, data, var_name=var_name)

    cells = split_cells(path)
    targets = indexes if indexes is not None else range(len(cells))
    for i in targets:
        title, code = cells[i]
        print(f"\n--- セル{i}: {title} ---")
        run_cell(kc, code)


if __name__ == "__main__":
    # 使い方:
    #   python test.py my_script.py                              → 全セル実行
    #   python test.py my_script.py 0 2                          → セル0とセル2だけ実行
    #   python test.py my_script.py --data '{"a": 1}'            → 辞書を input_data として注入して全セル実行
    #   python test.py my_script.py --data '{"a": 1}' --var cfg  → 辞書を cfg という変数名で注入
    #   python test.py my_script.py --data '{"a": 1}' 0 2        → データ注入後にセル0と2だけ実行
    args = sys.argv[1:]
    if not args:
        print("使い方: python test.py <ファイルパス> [--data 'JSON文字列'] [--var 変数名] [セル番号 ...]")
        sys.exit(1)

    path = args[0]
    rest = args[1:]

    data = None
    var_name = "input_data"
    cell_args = []
    i = 0
    while i < len(rest):
        if rest[i] == "--data":
            data = json.loads(rest[i + 1])
            i += 2
        elif rest[i] == "--var":
            var_name = rest[i + 1]
            i += 2
        else:
            cell_args.append(rest[i])
            i += 1

    nums = [int(n) for n in cell_args] if cell_args else None

    kc = connect()
    try:
        run_file(kc, path, nums, data=data, var_name=var_name)
    finally:
        kc.stop_channels()