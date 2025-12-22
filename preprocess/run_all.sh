#!/bin/bash

# ==============================================================================
# 0. CONFIGURATION (全局配置)
# ==============================================================================
BASE_DIR="/data0/determined/users/andywu/Audio-CALM-v2"
DATA_ROOT="$BASE_DIR/data"
RAW_ROOT="$DATA_ROOT/raw"
LATENT_ROOT="$DATA_ROOT/latents"
JSONL_ROOT="$DATA_ROOT/jsonl"

PYTHON_BIN="/data0/determined/users/andywu/config/.conda/envs/qwen2_CALM/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then PYTHON_BIN="python"; fi

VAE_CKPT="$BASE_DIR/outputs/checkpoints/audio_vae_4x_kl_annealing_l1_ssim/checkpoint-6900"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# === 核心开关 ===
DO_PIPELINE=true   # 下载 + 处理
DO_JSONL=true      # 生成索引

# === 数据集选择 ===
DATASETS_TO_RUN=(
    "librispeech"  # ASR
    "libritts"     # TTS
    "commonvoice"  # CV
)

# === Common Voice Credentials ===
CV_API_KEY="30deea4ea405c99c58e9dfac3d94243934ec5c26dfa510451003763f6978482b" 
CV_DOWNLOAD_TOKEN="dlt_f64dba7f-5087-4125-83ce-001365c6b59b" 

# ==============================================================================
# 工具函数
# ==============================================================================
contains_element() {
  local e match="$1"; shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

fast_download() {
    local url="$1"
    local filename="$2"
    if command -v aria2c &> /dev/null; then
        echo "🚀 [Aria2] Downloading $filename..."
        aria2c -x 16 -s 16 -c -o "$filename" "$url"
    else
        echo "🐢 [Wget] Downloading $filename..."
        wget -c -O "$filename" "$url"
    fi
}

# [核心升级] 极速检查逻辑
check_latents_exist() {
    local target_dir="$1"
    
    # 1. 最快检查：看有没有 .done 标记文件
    if [ -f "$target_dir/.done" ]; then
        return 0 # 存在且已完成
    fi

    # 2. 兜底检查：看有没有 .pt 文件 (防止手动删了 .done 但数据还在)
    if [ -d "$target_dir" ]; then
        # find -quit: 找到第一个就退出，极大加速
        if [ -n "$(find "$target_dir" -name "*.pt" -print -quit)" ]; then
            # 提示用户数据存在但没 .done
            # echo "   (Found .pt files but no .done marker. Will attempt resume.)"
            return 1 # 返回 False 让 Python 脚本去断点续传，并在完成后生成 .done
        fi
    fi
    return 1 # 不存在
}

# ==============================================================================
# 数据集处理函数
# ==============================================================================

# --- LibriSpeech ---
run_librispeech() {
    echo "========================================================"
    echo "🟢 [LibriSpeech] Checking Pipeline..."
    echo "========================================================"
    
    declare -A SUBSETS=(
        ["train-clean-100"]=1 ["train-clean-360"]=1 ["train-other-500"]=1
        ["dev-clean"]=1 ["test-clean"]=1
    )
    declare -A SPLIT_MAP=(
        ["train-clean-100"]="train" ["train-clean-360"]="train" ["train-other-500"]="train"
        ["dev-clean"]="dev" ["test-clean"]="test"
    )
    BASE_URL="https://www.openslr.org/resources/12"
    WORK_DIR="$RAW_ROOT/LibriSpeech"
    mkdir -p "$WORK_DIR"; cd "$WORK_DIR" || return

    for subset in "${!SUBSETS[@]}"; do
        split=${SPLIT_MAP[$subset]}
        OUT_DIR="$LATENT_ROOT/$split/LibriSpeech/$subset"

        # [加速] 检查是否已完成
        if check_latents_exist "$OUT_DIR"; then
            echo "✅ [Skip] $subset is marked done or populated."
            continue
        fi

        tar_file="${subset}.tar.gz"
        # 1. 下载
        if [ ! -d "$subset" ]; then
            if [ -f "$tar_file" ]; then
                echo "📦 Found tarball '$tar_file', extracting..."
                tar -xzf "$tar_file" --strip-components=1 && rm "$tar_file"
            else
                echo "⬇️  Downloading $subset..."
                fast_download "$BASE_URL/$tar_file" "$tar_file"
                if [ -f "$tar_file" ]; then
                    echo "📦 Extracting $subset..."
                    tar -xzf "$tar_file" --strip-components=1 && rm "$tar_file"
                fi
            fi
        fi

        # 2. 处理
        if [ -d "$subset" ]; then
            echo "⚙️  Processing $subset -> Latents..."
            $PYTHON_BIN "$BASE_DIR/preprocess/process_dataset.py" \
                --dataset_name "librispeech" --in_dir "$subset" --out_dir "$OUT_DIR" --vae_ckpt "$VAE_CKPT" --workers_per_gpu 4
            
            # [关键] 处理成功后，生成 .done 标记
            if [ $? -eq 0 ]; then
                touch "$OUT_DIR/.done"
                echo "✨ Marked $subset as done."
            fi
        fi
    done
    echo "✅ [LibriSpeech] Finished."
    echo ""
}

# --- LibriTTS-R ---
run_libritts() {
    echo "========================================================"
    echo "🔵 [LibriTTS-R] Checking Pipeline..."
    echo "========================================================"
    
    declare -A MAP=(
        ["train-clean-100"]="train_clean_100" ["train-clean-360"]="train_clean_360"
        ["train-other-500"]="train_other_500" ["dev-clean"]="dev_clean" ["test-clean"]="test_clean"
    )
    declare -A SPLIT_MAP=(
        ["train-clean-100"]="train" ["train-clean-360"]="train" ["train-other-500"]="train"
        ["dev-clean"]="dev" ["test-clean"]="test"
    )
    BASE_URL="https://www.openslr.org/resources/141"
    WORK_DIR="$RAW_ROOT/LibriTTS_R"
    mkdir -p "$WORK_DIR"; cd "$WORK_DIR" || return

    for subset in "${!MAP[@]}"; do
        dl_name=${MAP[$subset]}
        tar_file="${dl_name}.tar.gz"
        split=${SPLIT_MAP[$subset]}
        OUT_DIR="$LATENT_ROOT/$split/LibriTTS_R/$subset"

        # [加速] 检查是否已完成
        if check_latents_exist "$OUT_DIR"; then
            echo "✅ [Skip] $subset is marked done or populated."
            continue
        fi

        # 1. 下载
        if [ ! -d "$subset" ]; then
            if [ -d "$dl_name" ]; then mv "$dl_name" "$subset"; 
            elif [ -f "$tar_file" ]; then
                echo "📦 Found tarball '$tar_file', extracting..."
                tar -xzf "$tar_file" --strip-components=1 && rm "$tar_file"
                if [ -d "$dl_name" ] && [ "$dl_name" != "$subset" ]; then mv "$dl_name" "$subset"; fi
            else
                echo "⬇️  Downloading $subset..."
                fast_download "$BASE_URL/$tar_file" "$tar_file"
                if [ -f "$tar_file" ]; then
                    echo "📦 Extracting $subset..."
                    tar -xzf "$tar_file" --strip-components=1 && rm "$tar_file"
                    if [ -d "$dl_name" ] && [ "$dl_name" != "$subset" ]; then mv "$dl_name" "$subset"; fi
                fi
            fi
        fi

        # 2. 处理
        if [ -d "$subset" ]; then
            echo "⚙️  Processing $subset -> Latents..."
            $PYTHON_BIN "$BASE_DIR/preprocess/process_dataset.py" \
                --dataset_name "libritts" --in_dir "$subset" --out_dir "$OUT_DIR" --vae_ckpt "$VAE_CKPT" --workers_per_gpu 4
            
            # [关键] 标记完成
            if [ $? -eq 0 ]; then
                touch "$OUT_DIR/.done"
                echo "✨ Marked $subset as done."
            fi
        fi
    done
    echo "✅ [LibriTTS-R] Finished."
    echo ""
}

# --- Common Voice ---
run_commonvoice() {
    echo "========================================================"
    echo "🟣 [CommonVoice] Checking Pipeline..."
    echo "========================================================"
    
    CV_OUT="$LATENT_ROOT/train/CommonVoice"
    if check_latents_exist "$CV_OUT"; then
        echo "✅ [Skip] CommonVoice is marked done or populated."
        return
    fi

    WORK_DIR="$RAW_ROOT/CommonVoice"
    mkdir -p "$WORK_DIR"; cd "$WORK_DIR" || return
    CV_TAR="common_voice_en.tar.gz"

    if [ ! -d "clips" ] || [ ! -f "train.tsv" ]; then
        if [ ! -f "$CV_TAR" ]; then
            echo "🔍 Resolving real download URL via Python..."
            
            # 1. 调用 Python 脚本获取真实链接
            # 确保 get_cv_link.py 里的 Token 是最新的！
            REAL_URL=$($PYTHON_BIN "$BASE_DIR/preprocess/get_cv_link.py")
            
            if [ $? -eq 0 ] && [ -n "$REAL_URL" ]; then
                echo "✅ URL Resolved! Target: AWS S3"
                echo "🚀 [Aria2] Downloading (16 threads)..."
                # S3 链接带签名，不需要 Token，aria2c 可以跑满带宽
                aria2c -x 16 -s 16 -c -o "$CV_TAR" "$REAL_URL"
            else
                echo "❌ Failed to resolve URL. Using reliable Wget fallback..."
                # 最后的保底：如果 Python 解析失败，用 wget (虽然单线程但极其稳定)
                # wget 能正确处理 Auth 头和重定向
                wget -c --content-disposition --header="Authorization: Bearer $CV_API_KEY" \
                     "https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p1w0075nxxbe8bedl00/download/$CV_DOWNLOAD_TOKEN" \
                     -O "$CV_TAR"
            fi
        fi

        # 2. 校验与解压
        if [ -f "$CV_TAR" ]; then
            echo "📦 Verifying archive..."
            if ! gzip -t "$CV_TAR" &>/dev/null; then
                echo "❌ Error: Invalid gzip file. Deleting..."
                rm "$CV_TAR"
            else
                echo "📦 Extracting..."
                tar -xzf "$CV_TAR"
                FOUND_CLIPS=$(find . -maxdepth 3 -type d -name "clips" | head -n 1)
                if [ -n "$FOUND_CLIPS" ]; then
                    PARENT_DIR=$(dirname "$FOUND_CLIPS")
                    if [ "$PARENT_DIR" != "." ]; then mv "$PARENT_DIR"/* .; rmdir "$PARENT_DIR" 2>/dev/null || true; fi
                fi
                rm "$CV_TAR"
            fi
        fi
    fi

    # 3. 处理
    if [ -d "clips" ] && [ -f "train.tsv" ]; then
        echo "⚙️  Processing CommonVoice -> Latents..."
        $PYTHON_BIN "$BASE_DIR/preprocess/process_dataset.py" \
            --dataset_name "commonvoice" --in_dir "clips" --out_dir "$CV_OUT" --vae_ckpt "$VAE_CKPT" --cv_tsv "train.tsv" --workers_per_gpu 8
        if [ $? -eq 0 ]; then touch "$CV_OUT/.done"; echo "✨ Marked CommonVoice as done."; fi
    else
        echo "⚠️  CommonVoice raw data not found."
    fi
    echo "✅ [CommonVoice] Finished."
    echo ""
}

# ==============================================================================
# 主执行逻辑
# ==============================================================================

if [ "$DO_PIPELINE" = true ]; then
    echo "🚀 Starting Pipelines..."
    
    if contains_element "librispeech" "${DATASETS_TO_RUN[@]}"; then run_librispeech; fi
    if contains_element "libritts" "${DATASETS_TO_RUN[@]}"; then run_libritts; fi
    if contains_element "commonvoice" "${DATASETS_TO_RUN[@]}"; then run_commonvoice; fi
    
    echo "🎉 All pipelines finished!"
fi

if [ "$DO_JSONL" = true ]; then
    echo "========================================================"
    echo "📝 [Stage 3] Building JSONL Manifests..."
    echo "========================================================"
    mkdir -p $JSONL_ROOT
    $PYTHON_BIN "$BASE_DIR/preprocess/build_manifest.py" --latent_dir "$LATENT_ROOT/train" --output_file "$JSONL_ROOT/train.jsonl"
    $PYTHON_BIN "$BASE_DIR/preprocess/build_manifest.py" --latent_dir "$LATENT_ROOT/dev" --output_file "$JSONL_ROOT/dev.jsonl"
    $PYTHON_BIN "$BASE_DIR/preprocess/build_manifest.py" --latent_dir "$LATENT_ROOT/test" --output_file "$JSONL_ROOT/test.jsonl"
    echo "✅ JSONL generation complete!"
fi