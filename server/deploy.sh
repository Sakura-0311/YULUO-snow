#!/usr/bin/env bash
# YULUO-snow 限时推理服务 —— 通用部署脚本
#
# 适用：Ubuntu 22.04/24.04，2 核 4 线程 / 8G 内存 / 无 GPU（CPU 推理）
# 形态：服务只绑 127.0.0.1，由已有 Caddy 在 443 上反代，复用现成证书。
#
# 用法：
#   sudo DOMAIN=example.com bash deploy.sh
#
# 前置：
#   1. 目标域名已解析到本机，且已有 Caddy 在跑（复用其 Let's Encrypt 证书）
#   2. 本脚本会把 Caddyfile 备份成 Caddyfile.bak-<时间戳>，校验失败自动回滚
set -euo pipefail

DOMAIN="${DOMAIN:?请指定 DOMAIN，例如 DOMAIN=example.com sudo -E bash deploy.sh}"
ROOT="${ROOT:-/opt/yuluo}"
SUBPATH="${SUBPATH:-yuluo}"
PORT_API="${PORT_API:-8000}"
PORT_BACKEND="${PORT_BACKEND:-8081}"
PKG_ZIP="${PKG_ZIP:-$(cd "$(dirname "$0")/.." && pwd)/../YULUO-snow.zip}"
LOG=/var/log/yuluo-deploy.log
STAMP=$(date +%Y%m%d-%H%M)
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { echo "错误: $*"; exit 1; }

say "0/8 前置检查"
[ "$(id -u)" -eq 0 ] || die "请用 sudo 运行"
[ -f "$PKG_ZIP" ] || die "找不到交付包: $PKG_ZIP（可用 PKG_ZIP=... 指定）"
command -v caddy >/dev/null || die "未安装 Caddy，本脚本依赖它做 443 反代"
systemctl is-active --quiet caddy || die "Caddy 未运行"
free -m | sed -n '1,2p'; df -h / | tail -1; echo "nproc=$(nproc)"

say "1/8 swap（fp16 合并峰值约 3.5G，无 swap 有 OOM 风险）"
if ! swapon --show | grep -q .; then
  fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile; mkswap /swapfile >/dev/null; swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "swap 已启用"
else
  echo "已有 swap，跳过"
fi

say "2/8 系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv cmake build-essential unzip git >/dev/null

say "3/8 解包交付物"
rm -rf "$ROOT"; mkdir -p "$ROOT"
unzip -q -o "$PKG_ZIP" -d "$ROOT"
SRC=$(find "$ROOT" -maxdepth 2 -type d -name 'YULUO-snow' | head -1)
[ -n "$SRC" ] || die "包内没有 YULUO-snow/ 目录"
mv "$SRC" "$ROOT/app"

say "4/8 Python 依赖（CPU torch）"
python3 -m venv "$ROOT/venv"
PIP="$ROOT/venv/bin/pip"; PY="$ROOT/venv/bin/python"
"$PIP" install -q --upgrade pip wheel
"$PIP" install -q --index-url https://download.pytorch.org/whl/cpu torch
"$PIP" install -q transformers peft accelerate safetensors huggingface_hub \
    fastapi 'uvicorn[standard]' pydantic jsonschema sentencepiece gguf kernels \
    'bitsandbytes>=0.46.1'
"$PY" -c "import torch,transformers,peft,fastapi;print('torch',torch.__version__)"

say "5/8 合并 LoRA 进反量化 4-bit 权重（关键，见 docs/FINDINGS.md 发现 2）"
"$PY" "$ROOT/app/tools/merge_into_dequant.py"

say "6/8 转 GGUF 并量化 Q8_0"
cd "$ROOT"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF >/dev/null
cmake --build build -j"$(nproc)" --target llama-quantize llama-server >/dev/null
"$PY" convert_hf_to_gguf.py "$ROOT/app/_merged4deq" --outfile "$ROOT/model-f16.gguf" --outtype f16
./build/bin/llama-quantize "$ROOT/model-f16.gguf" "$ROOT/model-q8.gguf" Q8_0
rm -f "$ROOT/model-f16.gguf"; rm -rf "$ROOT/app/_merged4deq"

say "7/8 systemd：后端 + API"
cat > /etc/systemd/system/yuluo-llama.service <<EOF
[Unit]
Description=YULUO-snow llama.cpp backend (Q8_0)
After=network.target

[Service]
Type=simple
User=${SUDO_USER:-ubuntu}
ExecStart=$ROOT/llama.cpp/build/bin/llama-server -m $ROOT/model-q8.gguf \\
  --host 127.0.0.1 --port $PORT_BACKEND -t 3 -c 2048 --no-webui
Restart=always
RestartSec=3
Nice=10
MemoryMax=4G

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/yuluo-api.service <<EOF
[Unit]
Description=YULUO-snow limited API
After=yuluo-llama.service

[Service]
Type=simple
User=${SUDO_USER:-ubuntu}
WorkingDirectory=$ROOT/app/server
Environment=APP=$ROOT/app
Environment=YULUO_BACKEND=http://127.0.0.1:$PORT_BACKEND/v1/chat/completions
ExecStart=$ROOT/venv/bin/uvicorn app:app --host 127.0.0.1 --port $PORT_API --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
chown -R "${SUDO_USER:-ubuntu}" "$ROOT"
systemctl daemon-reload
systemctl enable --now yuluo-llama.service yuluo-api.service
sleep 4
systemctl is-active yuluo-llama yuluo-api

say "8/8 Caddy 反代（必须插在兜底 handle 之前，且不经过 forward_auth）"
CADDY=/etc/caddy/Caddyfile
cp "$CADDY" "$CADDY.bak-$STAMP"
if grep -q "handle_path /$SUBPATH/\*" "$CADDY"; then
  echo "已存在 /$SUBPATH/* 规则，跳过"
else
  "$PY" - "$CADDY" "$DOMAIN" "$SUBPATH" "$PORT_API" <<'PYEOF'
import sys
path, domain, sub, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(path, encoding="utf-8").read()
idx = s.find(domain)
if idx < 0:
    print("Caddyfile 里找不到域名 " + domain + "，请手工添加规则"); sys.exit(2)
# 优先插到该站点块的兜底 handle 之前
rel = s.find("\n\thandle {\n", idx)
if rel > 0:
    ins = rel + 1
else:
    end = s.find("\n}", idx)
    if end < 0:
        print("站点块闭合无法定位"); sys.exit(3)
    ins = end
block = ("\t# YULUO-snow 推理 API（独立 handle，绕过 forward_auth）\n"
         "\thandle_path /" + sub + "/* {\n"
         "\t\treverse_proxy 127.0.0.1:" + port + "\n"
         "\t}\n\n")
s = s[:ins] + block + s[ins:]
open(path, "w", encoding="utf-8").write(s)
print("已插入 /" + sub + "/* -> 127.0.0.1:" + port)
PYEOF
  if ! caddy validate --config "$CADDY" 2>&1 | tail -2; then
    echo "Caddyfile 校验失败，回滚"; cp "$CADDY.bak-$STAMP" "$CADDY"; die "请手工配置反代"
  fi
  systemctl reload caddy
fi

say "验收"
sleep 6
echo "[本地]";   curl -fsS "http://127.0.0.1:$PORT_API/v1/health" || echo "本地未就绪"
echo; echo "[公网]"; curl -fsS "https://$DOMAIN/$SUBPATH/v1/health" || echo "公网未就绪"
echo; echo "[推理]"
time curl -sS -X POST "https://$DOMAIN/$SUBPATH/v1/extract" \
  -H 'content-type: application/json' \
  -d '{"text":"会员首月 9.9 元，次月起每月 29 元自动续费，可随时取消，已付费用不予退还。","source_id":"deploy_smoke"}' \
  | head -c 600
echo
say "完成。日志 $LOG"
