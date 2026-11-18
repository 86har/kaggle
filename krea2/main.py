# %%
import os, glob, subprocess, shutil, errno, re, uuid, time, random, json, gc, struct, sys, inspect
import requests

CIVITAI_TOKEN = tokens["CIVITAI"]
MEGA_EMAIL = tokens["MEGA_EMAIL"]
MEGA_PASS  = tokens["MEGA_PASS"]

# ---------------------------------------------------------------
# 保存先の自動選択
# /kaggle/working は上限 約20GB(しかもセッション保存の対象)なので避け、
# 空きに余裕のある /kaggle/temp か /tmp を優先して使う。
# 手動で決めたいときは BASE_DIR_OVERRIDE に path を入れる。
# ---------------------------------------------------------------
BASE_DIR_OVERRIDE = None
MIN_FREE_GB = 15

def free_gb(path):
    p = path
    while not os.path.exists(p):
        p = os.path.dirname(p) or "/"
    return shutil.disk_usage(p).free / 1e9

def pick_base():
    if BASE_DIR_OVERRIDE:
        return BASE_DIR_OVERRIDE
    cands = ["/kaggle/temp", "/tmp", "/kaggle/working"]
    ok = [c for c in cands if os.path.isdir(c) and os.access(c, os.W_OK)]
    for c in ok:
        if free_gb(c) >= MIN_FREE_GB:
            return c
    # どこも足りない場合は一番空きが多い所
    return max(ok, key=free_gb)

BASE = pick_base()
WORK = f"{BASE}/work"
os.makedirs(WORK, exist_ok=True)

# HuggingFaceのキャッシュも同じ場所へ(ホームの容量を食わせない)
os.environ["HF_HOME"] = f"{WORK}/hf_cache"
os.environ["PIP_NO_CACHE_DIR"] = "1"

COMFY = f"{WORK}/ComfyUI"
M = f"{COMFY}/models"

# 使うチェックポイント(Krea 2 系)
CKPT_NAME = "lustify-v10-krea-turbo-fp8.safetensors"
CKPT = f"{M}/diffusion_models/{CKPT_NAME}"   # リンクではなく直接ここに置く(二重取りを避ける)

# Krea 2 の必須ファイル(Flux 用の T5 / CLIP / ae.safetensors は使えません)
TE_NAME  = "qwen3vl_4b_fp8_scaled.safetensors"   # CLIPLoader の type は krea2
VAE_NAME = "qwen_image_vae.safetensors"

OUT_DIR   = f"{COMFY}/output"
STAGE_DIR = f"{WORK}/upload_stage"   # アップロード失敗時の退避先
MEGA_DEST = "/outputs/Echs/"         # 末尾の / でフォルダとして扱う

for d in ["diffusion_models", "text_encoders", "vae", "loras"]:
    os.makedirs(f"{M}/{d}", exist_ok=True)

print(f"保存先: {BASE}  (空き {free_gb(BASE):.1f} GB)")

# megacmd のインストール(.debは入れたら即削除)
!wget -q https://mega.nz/linux/repo/xUbuntu_22.04/amd64/megacmd-xUbuntu_22.04_amd64.deb -O /tmp/megacmd.deb && apt-get install -y -qq /tmp/megacmd.deb; rm -f /tmp/megacmd.deb

import numpy as np
from PIL import Image

RAM_DIR = "/dev/shm/upload_tmp" if os.path.isdir("/dev/shm") else f"{WORK}/upload_tmp"


# ---------------------------------------------------------------
# 容量まわりのユーティリティ
# ---------------------------------------------------------------
def disk_report(top=10):
    """容量の状況と、大きいファイルの一覧を表示する"""
    print(f"[{BASE}] 空き {free_gb(BASE):.1f} GB")
    files = []
    for root, _, names in os.walk(WORK):
        for n in names:
            p = os.path.join(root, n)
            try:
                if not os.path.islink(p):
                    files.append((os.path.getsize(p), p))
            except OSError:
                pass
    for size, p in sorted(files, reverse=True)[:top]:
        print(f"  {size/1e9:6.2f} GB  {p}")

def cleanup_space():
    """消しても安全なものだけ削除する(未アップロードの画像は消さない)"""
    freed = 0
    # 途中で失敗したダウンロード
    for p in glob.glob(f"{WORK}/**/*.part", recursive=True):
        try:
            freed += os.path.getsize(p); os.remove(p)
        except OSError:
            pass
    # HFのキャッシュ・一時ダウンロード
    for d in [f"{WORK}/hf_cache", f"{WORK}/hf_tmp"]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    # pipキャッシュ、megacmdのdeb
    shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)
    if os.path.exists("/tmp/megacmd.deb"):
        os.remove("/tmp/megacmd.deb")
    gc.collect()
    print(f"掃除完了。空き {free_gb(BASE):.1f} GB")

def ensure_space(path, need_gb, label=""):
    """need_gb 必要。足りなければ掃除して再確認、それでも足りなければ分かりやすく止める"""
    d = os.path.dirname(path) or "."
    if free_gb(d) >= need_gb:
        return
    cleanup_space()
    if free_gb(d) < need_gb:
        raise RuntimeError(
            f"{label} 容量不足: 必要 約{need_gb:.2f} GB / 空き {free_gb(d):.2f} GB\n"
            f"disk_report() で大きいファイルを確認し、不要なLoRA等を delete_lora() で消してください。"
        )

def _scrub(msg, *secrets):
    for s in secrets:
        if s:
            msg = msg.replace(s, "***")
    return msg

def download_file(url, dest, token=None, label="DOWNLOAD", margin_gb=1.0, force=False):
    """
    容量チェック付きダウンロード。
    - 開始前に content-length と空き容量を比較
    - 書き込み中の ENOSPC(容量切れ)を検出して .part を確実に削除
    - 完了後にサイズ不一致(途中切れ)を検出
    """
    if os.path.exists(dest) and not force:
        print(f"既にあります。ダウンロード不要: {dest}")
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with requests.get(url, headers=headers, stream=True,
                          allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            cd = r.headers.get("content-disposition", "")
            m = re.search(r'filename="?([^";]+)"?', cd)
            print("配信側のファイル名:", m.group(1) if m else "(不明)")
            total = int(r.headers.get("content-length", 0))

            if total:
                ensure_space(dest, total / 1e9 + margin_gb, label)
            else:
                ensure_space(dest, 3.0, label)   # サイズ不明なら最低3GBは要求

            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r[{label}] {downloaded/1e9:.2f} / {total/1e9:.2f} GB", end="")

        if total and downloaded != total:
            raise RuntimeError(f"サイズ不一致(途中で切れた可能性): {downloaded} / {total} bytes")
        os.replace(tmp, dest)
        print("\n保存先:", dest)

    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.errno == errno.ENOSPC:
            raise RuntimeError(f"{label} 失敗: ディスク容量が足りません。disk_report() / cleanup_space() を実行してください。") from None
        raise RuntimeError(f"{label} 失敗: {type(e).__name__}: {_scrub(str(e), token)}") from None
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"{label} 失敗: {type(e).__name__}: {_scrub(str(e), token)}") from None

    return dest


# ---------------------------------------------------------------
# MEGAへのアップロード(未アップロードの退避画像も一緒に送る)
# ---------------------------------------------------------------
def upload_images_to_mega(images):
    """images: VAEDecodeの出力 (B, H, W, 3) のtorchテンソル"""
    paths = []
    for img in images:
        arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        # 1) RAM領域に書く。溢れたらディスクへ
        for d in (RAM_DIR, STAGE_DIR):
            try:
                os.makedirs(d, exist_ok=True)
                p = f"{d}/{name}"
                Image.fromarray(arr).save(p)
                paths.append(p)
                break
            except OSError as e:
                if os.path.exists(f"{d}/{name}"):
                    os.remove(f"{d}/{name}")
                if e.errno != errno.ENOSPC:
                    raise
        else:
            raise RuntimeError("画像の一時保存先がどこも満杯です。cleanup_space() を実行してください。")

    # 前回失敗して退避された画像があれば、今回まとめて送る
    stashed = [p for p in glob.glob(f"{STAGE_DIR}/*.png") if p not in paths]
    all_paths = paths + stashed

    # 2) すぐアップロード
    ok = False
    try:
        subprocess.run(["mega-login", MEGA_EMAIL, MEGA_PASS], check=True, capture_output=True)
        subprocess.run(["mega-put", "-c", *all_paths, MEGA_DEST], check=True, capture_output=True)
        ok = True
        print(f"{len(all_paths)}枚アップロード完了(うち退避分 {len(stashed)}枚) → {MEGA_DEST}")
    except subprocess.CalledProcessError as e:
        # 失敗しても画像が消えないよう、ディスクに退避
        os.makedirs(STAGE_DIR, exist_ok=True)
        for p in paths:
            if os.path.dirname(p) != STAGE_DIR and os.path.exists(p):
                try:
                    shutil.move(p, f"{STAGE_DIR}/{os.path.basename(p)}")
                except OSError:
                    print("退避にも失敗(容量不足): ", os.path.basename(p))
        err = (e.stderr or b"").decode(errors="ignore")[-300:]
        err = _scrub(err, MEGA_EMAIL, MEGA_PASS)
        raise RuntimeError(f"MEGAアップロード失敗(画像は{STAGE_DIR}に退避): {err}") from None
    finally:
        subprocess.run(["mega-logout"], check=False, capture_output=True)
        if ok:
            for p in all_paths:
                if os.path.exists(p):
                    os.remove(p)
        else:
            # 失敗時はRAM領域の分だけ消す(退避済みなのでOK)
            for p in paths:
                if p.startswith(RAM_DIR) and os.path.exists(p):
                    os.remove(p)

print("CKPT:", CKPT, "/ 存在:", os.path.exists(CKPT))

# %%
print("ComfyUI Installing...")
if not os.path.exists(f"{COMFY}/main.py"):
    ensure_space(COMFY, 3.0, "ComfyUI")
    # models/ などが先に作られていても失敗しないよう、別の場所にcloneしてから重ねる
    tmp_clone = f"{WORK}/ComfyUI_clone"
    shutil.rmtree(tmp_clone, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI", tmp_clone], check=True)
    shutil.copytree(tmp_clone, COMFY, dirs_exist_ok=True)   # 既存のmodels/やダウンロード済みファイルは残る
    shutil.rmtree(tmp_clone, ignore_errors=True)
    subprocess.run(["pip", "install", "-q", "--no-cache-dir", "-r", f"{COMFY}/requirements.txt"], check=True)
print("ComfyUI OK:", os.path.exists(f"{COMFY}/main.py"))

# %%
MODEL_VERSION_ID = "3112728"
ckpt_url = f"https://civitai.com/api/download/models/{MODEL_VERSION_ID}"

download_file(ckpt_url, CKPT, token=CIVITAI_TOKEN, label="CKPT")

def download_lora(lora_name, version_id, token=None, force=False):
    """
    Civitaiから LoRA をダウンロードして M/loras/ に保存する。
    lora_name: 保存するファイル名 (例: "my_lora_v1.safetensors")
    version_id: Civitai の Model Version ID
    戻り値: 保存先のフルパス
    """
    token = token or CIVITAI_TOKEN
    path = f"{M}/loras/{lora_name}"
    url = f"https://civitai.com/api/download/models/{version_id}"
    return download_file(url, path, token=token, label="LORA", margin_gb=0.5, force=force)

def delete_lora(lora_name):
    """使わなくなったLoRAを削除して容量を空ける"""
    p = f"{M}/loras/{lora_name}"
    if os.path.exists(p):
        size = os.path.getsize(p)
        os.remove(p)
        print(f"削除: {lora_name} ({size/1e9:.2f} GB)  空き {free_gb(BASE):.1f} GB")
    else:
        print("見つかりません:", p)

# %%
from huggingface_hub import HfApi, hf_hub_download

assert os.path.exists(CKPT), "チェックポイントがありません。セル 3 を確認してください。"

# Qwen3-VL エンコーダーと Qwen VAE(未取得のものだけ)
wanted = {TE_NAME: "text_encoders", VAE_NAME: "vae"}
for name, folder in list(wanted.items()):
    if os.path.exists(f"{M}/{folder}/{name}"):
        del wanted[name]

if wanted:
    ensure_space(f"{M}/text_encoders/x", 6.0, "Text Encoder/VAE")
    for repo in ["Comfy-Org/Krea-2", "Comfy-Org/Krea2"]:
        try:
            files = HfApi().list_repo_files(repo)
        except Exception as e:
            print(repo, "→ 取得失敗:", type(e).__name__)
            continue
        for name, folder in list(wanted.items()):
            hit = [f for f in files if f.endswith("/" + name) or f == name]
            if hit:
                tmp_dir = f"{WORK}/hf_tmp"
                try:
                    # local_dir に直接落とし、キャッシュとの二重保存を避ける
                    got = hf_hub_download(repo, hit[0], local_dir=tmp_dir)
                    shutil.move(got, f"{M}/{folder}/{name}")   # コピーではなく移動
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    shutil.rmtree(f"{WORK}/hf_cache", ignore_errors=True)
                print("取得:", repo, hit[0])
                del wanted[name]
        if not wanted:
            break

assert not wanted, f"取得できませんでした: {list(wanted)}"

# ダミーLoRA(空のsafetensors)
dummy = f"{M}/loras/dummy_none.safetensors"
if not os.path.exists(dummy):
    with open(dummy, "wb") as f:
        f.write(struct.pack("<Q", 2) + b"{}")

for f in [f"diffusion_models/{CKPT_NAME}", f"text_encoders/{TE_NAME}", f"vae/{VAE_NAME}", "loras/dummy_none.safetensors"]:
    print("OK  " if os.path.exists(f"{M}/{f}") else "なし", f)

disk_report()

# %%
sys.path.insert(0, COMFY)
os.chdir(COMFY)  # folder_paths などが相対パスで output/ 等を解決するため
sys.argv = [sys.argv[0]]  # Jupyterのカーネル引数をComfyUIのargparseに渡さないため

import nodes

r = nodes.init_extra_nodes()   # 新しい版は coroutine、古い版は同期で即完了
if inspect.isawaitable(r):
    await r

NCM = nodes.NODE_CLASS_MAPPINGS
print("ノード読み込みOK。登録数:", len(NCM))

# %%
import torch

PERSIST_CLASSES = {"UNETLoader", "CLIPLoader", "VAELoader"}
LOADER_CACHE = {}   # 同じローダー+同じ入力なら、2回目以降は読み込み済みのモデルを返す

@torch.inference_mode()
def run_graph(wf):
    cache = {}

    def resolve(node_id):
        if node_id in cache:
            return cache[node_id]
        spec = wf[node_id]

        pkey = None
        if spec["class_type"] in PERSIST_CLASSES:
            pkey = (spec["class_type"], json.dumps(spec["inputs"], sort_keys=True))
            if pkey in LOADER_CACHE:
                cache[node_id] = LOADER_CACHE[pkey]
                return cache[node_id]

        cls = NCM[spec["class_type"]]
        inst = cls()
        inputs = {}
        for key, val in spec["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and val[0] in wf:
                out = resolve(val[0])
                inputs[key] = out[val[1]]
            else:
                inputs[key] = val
        fn = getattr(inst, inst.FUNCTION)
        result = fn(**inputs)

        if pkey:
            LOADER_CACHE[pkey] = result
        cache[node_id] = result
        return result

    for node_id in wf:
        resolve(node_id)
    return cache


def free_models():
    """モデルを入れ替えたいとき/終了時に呼ぶ"""
    import comfy.model_management as mm
    LOADER_CACHE.clear()
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def generate(prompt, seed=None, steps=8, cfg=1.0, w=1024, h=1024, n=1,
             loras=None):
    """
    loras: 例 [("my_lora_v1.safetensors", 0.8)] のようなリスト。
           複数指定するとモデル/CLIPに順番に適用される。
           None または [] ならLoRAなし。
    """
    print(prompt)
    seed = random.randint(0, 2**32 - 1) if seed is None else seed

    # 画像の退避先がほぼ満杯なら、生成前に警告
    if free_gb(BASE) < 0.5:
        print(f"⚠ 空きが {free_gb(BASE):.2f} GB しかありません。cleanup_space() / delete_lora() を検討してください。")

    wf = {
      "1": {"class_type": "UNETLoader", "inputs": {"unet_name": CKPT_NAME, "weight_dtype": "default"}},
      "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "krea2", "device": "default"}},
      "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
    }

    model_ref = ["1", 0]
    clip_ref  = ["2", 0]

    # LoRAを順番にチェーン適用
    if loras:
        for i, (lora_name, strength) in enumerate(loras):
            node_id = f"lora_{i}"
            wf[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_ref,
                    "clip": clip_ref,
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "strength_clip": strength,
                },
            }
            model_ref = [node_id, 0]
            clip_ref  = [node_id, 1]

    wf.update({
      "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}},
      "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
      "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": n}},
      "7": {"class_type": "KSampler", "inputs": {
            "model": model_ref, "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple",
            "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}},
      "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
    })

    t0 = time.time()
    result = run_graph(wf)
    images = result["8"][0]
    print(f"seed={seed}  {time.time()-t0:.0f}秒")

    upload_images_to_mega(images)
# %%
generate("""
A realistic photo of an 18-year-old Japanese woman with large breasts and fair, translucent skin.
She is sitting on a black fabric sofa with her legs slightly apart, smiling at the camera.
The camera captures her clearly from her thighs all the way up to her head.

The woman has large breasts, estimated to be an E-cup.

She is wearing an oversized white collared shirt with all the buttons undone.

The camera captures her from a slightly low angle,
and her pussy is faintly visible.

Her large, dark eyes are slightly glistening, giving her a sweet, coquettish look.
Her expression blends innocent cuteness with a touch of playfulness and a slightly mischievous, devilish charm.
She has long, straight black hair that reaches down to her chest,
styled in a cute look with sheer bangs above her eyes that let her forehead peek through slightly.

Her face is small and oval-shaped, with well-proportioned features that create a doll-like appearance with very little empty space.
She has double eyelids with a very wide crease; her large, dark eyes—which are slightly droopy—and her very large, plump tear troughs accentuate her small-animal-like cuteness.
Combined with her translucent, fair skin, she possesses a unique charm that blends a girlish, innocent fragility with a cool, mysterious allure.
She applies makeup to create natural shadows under her tear troughs.

The left side of the background is covered by brown curtains, while the right side is a gray wall.
Her skin looks soft and has a slightly warm tone.

This image was taken with a smartphone’s front-facing camera,
and a faint flash reflects off their skin in the dimly lit room.
The image quality is slightly grainy, giving it a realistic texture typical of photos taken with a smartphone’s front-facing camera.
""", n=1, w=880, h=1184)
