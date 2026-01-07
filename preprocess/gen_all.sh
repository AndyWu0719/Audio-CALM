set -e

ROOT=/data0/determined/users/andywu/Audio-CALM-v2
RAW_LIBRISPEECH=/data0/determined/users/andywu/Audio-CALM-v2/data/raw/LibriSpeech
RAW_LIBRITTS=/data0/determined/users/andywu/Audio-CALM-v2/data/raw/LibriTTS_R
VAE_CKPT=${ROOT}/outputs/checkpoints/vae_4x_128_5e-4/checkpoint-17350

OUT_MEL=${ROOT}/data/mels
OUT_LAT=${ROOT}/data/latents
mkdir -p "${OUT_MEL}" "${OUT_LAT}"

collect_subdirs() {
  local base=$1
  local split=$2
  if [ -d "${base}/${split}" ]; then
    find "${base}/${split}" -maxdepth 1 -mindepth 1 -type d | sort
  else
    # 情况2：直接在 base 下就是 train-clean-100 这类
    find "${base}" -maxdepth 1 -mindepth 1 -type d -name "${split}*" | sort
  fi
}

run_proc() {
  local ds_dir=$1          # 输出目录名：LibriSpeech / LibriTTS_R
  local ds_key=$2          # 传给 process_dataset 的名称：librispeech / libritts
  local split=$3           # train / dev / test
  local subset_path=$4     # 原始子目录全路径
  local mode=$5            # mels / latents
  local subset_name
  subset_name=$(basename "${subset_path}")
  local out_dir=${ROOT}/data/${mode}/${split}/${ds_dir}/${subset_name}
  echo "===> ${ds_dir} ${split}/${subset_name} ${mode} -> ${out_dir}"
  if [ "${mode}" = "mels" ]; then
    python preprocess/process_dataset.py \
      --dataset_name ${ds_key} \
      --in_dir "${subset_path}" \
      --out_dir "${out_dir}" \
      --mel_only \
      --num_gpus 4 --workers_per_gpu 2
  else
    python preprocess/process_dataset.py \
      --dataset_name ${ds_key} \
      --in_dir "${subset_path}" \
      --out_dir "${out_dir}" \
      --vae_ckpt "${VAE_CKPT}" \
      --num_gpus 4 --workers_per_gpu 2
  fi
}

build_manifest() {
  local ds_dir=$1  # LibriSpeech / LibriTTS_R
  local split=$2   # train / dev / test
  local root_dir=${ROOT}/data/latents/${split}/${ds_dir}
  local out_file=${root_dir}/manifest_latents.jsonl
  echo "===> manifest ${ds_dir} ${split}"
  python preprocess/build_manifest.py \
    --latent_dir "${root_dir}" \
    --output_file "${out_file}"
}

# 遍历三类 split
for split in train dev test; do
  # LibriSpeech 子目录
  for sub in $(collect_subdirs "${RAW_LIBRISPEECH}" "${split}*"); do
    [ -d "${sub}" ] || continue
    run_proc LibriSpeech librispeech ${split} "${sub}" mels
    run_proc LibriSpeech librispeech ${split} "${sub}" latents
  done
  # LibriTTS_R 子目录
  for sub in $(collect_subdirs "${RAW_LIBRITTS}" "${split}*"); do
    [ -d "${sub}" ] || continue
    run_proc LibriTTS_R libritts ${split} "${sub}" mels
    run_proc LibriTTS_R libritts ${split} "${sub}" latents
  done
done

# 生成 manifest（latents 用）
for ds_dir in LibriSpeech LibriTTS_R; do
  for split in train dev test; do
    build_manifest ${ds_dir} ${split}
  done
done

echo "All done."