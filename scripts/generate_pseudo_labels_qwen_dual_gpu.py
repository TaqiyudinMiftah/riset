from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

from PIL import Image
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emotion_reasoning.utils.image_ops import draw_red_box, load_rgb_image
from emotion_reasoning.utils.io import load_records, save_records
list_of_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# "Analyze the main person in the image. In a single, concise paragraph, describe their geometric facial features (e.g., 'flat lip line', 'narrowed eyes', 'lowered brow') and body posture. Then, describe the immediate environmental context. CRITICAL: Do NOT use any emotion-related words or adjectives (such as 'neutral', 'angry', 'relaxed', 'happy') to describe their expressions. Describe only the physical state."
DEFAULT_PROMPT_WITH_BOX_TEMPLATE = (
    f"Given the following list of emotions: {list_of_emotions}, please explain in detail which emotions are more suitable for describing how the person feels based on the image context. Analyze the main person in the image. In a single, concise paragraph, describe their geometric facial features and body posture. Then, describe the immediate environmental context."
)

DEFAULT_PROMPT_NO_BOX_TEMPLATE = DEFAULT_PROMPT_WITH_BOX_TEMPLATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen pseudo-labels with dual-GPU workers and a single JSONL checkpoint file."
    )
    parser.add_argument("--mode", choices=["launcher", "worker"], default="launcher")

    parser.add_argument(
        "--annotation-path",
        default=str(
            PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "annotations" / "caers_annotations_with_val.jsonl"
        ),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "caer_dataset" / "CAER-S"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "stage1_pseudo_labels_qwen.jsonl"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--local-model-dir",
        default=str(PROJECT_ROOT / "model_cache" / "qwen3_vl_4b_instruct"),
    )
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "model_cache"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--num-datasets", type=int, default=None)
    parser.add_argument("--class-names", default="")

    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--prompt-with-box-template", default=DEFAULT_PROMPT_WITH_BOX_TEMPLATE)
    parser.add_argument("--prompt-no-box-template", default=DEFAULT_PROMPT_NO_BOX_TEMPLATE)
    return parser.parse_args()


def _sample_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("sample_id", f"sample_{index:06d}"))


def _resolve_record_limit(sample_limit: int | None, num_datasets: int | None) -> int | None:
    if sample_limit is not None and num_datasets is not None:
        raise ValueError("Gunakan salah satu: --sample-limit atau --num-datasets, jangan keduanya.")

    record_limit = num_datasets if num_datasets is not None else sample_limit
    if record_limit is None:
        return None
    if int(record_limit) <= 0:
        raise ValueError("Jumlah data harus > 0.")
    return int(record_limit)


def _prepare_records(annotation_path: Path, record_limit: int | None) -> list[dict[str, Any]]:
    records = load_records(annotation_path)
    if record_limit is not None:
        records = records[:record_limit]
    return records


def _load_rows_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_records(path)


def _load_existing_map(output_path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    existing_by_id: dict[str, dict[str, Any]] = {}
    if not resume:
        return existing_by_id
    for row in _load_rows_if_exists(output_path):
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            existing_by_id[sid] = row
    return existing_by_id


def _first_missing(records: list[dict[str, Any]], completed_ids: set[str]) -> tuple[int | None, str | None]:
    for index, row in enumerate(records):
        sid = _sample_id(row, index)
        if sid not in completed_ids:
            return index, sid
    return None, None


def _ordered_global(records: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        sid = _sample_id(row, index)
        if sid in by_id:
            ordered.append(by_id[sid])
    return ordered


def _parse_class_names(class_names: str, records: list[dict[str, Any]]) -> list[str]:
    provided = [item.strip() for item in class_names.split(",") if item.strip()]
    if provided:
        return provided
    labels = {
        str(row.get("labels", "")).strip()
        for row in records
        if str(row.get("labels", "")).strip()
    }
    return sorted(labels)


def _render_prompt(template: str, emotion_list_text: str) -> str:
    try:
        return template.format(emotion_list=emotion_list_text)
    except KeyError as exc:
        raise ValueError(
            f"Template prompt mengandung placeholder tidak dikenal: {exc}. "
            "Gunakan {emotion_list} jika membutuhkan daftar kelas."
        ) from exc


def _resolve_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _local_model_ready(local_model_dir: Path) -> bool:
    if not local_model_dir.exists():
        return False
    has_config = (local_model_dir / "config.json").exists()
    has_weights = any(local_model_dir.glob("*.safetensors")) or any(local_model_dir.glob("pytorch_model*"))
    has_processor = (local_model_dir / "processor_config.json").exists() or (
        local_model_dir / "tokenizer_config.json"
    ).exists()
    return has_config and has_weights and has_processor


def _write_model_manifest(local_model_dir: Path, model_id: str, source: str) -> None:
    payload = {
        "model_id": model_id,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (local_model_dir / "model_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ensure_local_model(model_id: str, local_model_dir: Path, cache_dir: Path) -> None:
    local_model_dir.mkdir(parents=True, exist_ok=True)

    if _local_model_ready(local_model_dir):
        print(f"Model lokal sudah tersedia: {local_model_dir}")
        return

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub tidak tersedia. Install dependency ini untuk download model dari Hugging Face."
        )

    source = "downloaded"
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_model_dir),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        source = "hf_cache"
        print(f"Model ditemukan di Hugging Face cache lokal untuk {model_id}")
    except Exception:
        print(f"Model belum ada di cache, mulai download dari Hugging Face: {model_id}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_model_dir),
            cache_dir=str(cache_dir),
            local_files_only=False,
        )

    if not _local_model_ready(local_model_dir):
        raise RuntimeError(
            f"Model lokal tidak lengkap setelah proses prepare: {local_model_dir}. "
            "Cek koneksi atau ulangi download."
        )

    _write_model_manifest(local_model_dir, model_id=model_id, source=source)
    print(f"Model siap dipakai dan tersimpan di: {local_model_dir}")


def _load_qwen_model(model_source: str | Path, device: str, cache_dir: Path) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(device),
        "low_cpu_mem_usage": True,
        "cache_dir": str(cache_dir),
        "local_files_only": True,
    }

    if Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_source, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)

    model = model.to(device)
    model.eval()
    return model


def _generate_caption(
    model: torch.nn.Module,
    processor: AutoProcessor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        fallback_prompt = f"USER: <image>\\n{prompt_text}\\nASSISTANT:"
        inputs = processor(images=image, text=fallback_prompt, return_tensors="pt")

    inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if "input_ids" in inputs:
        output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _validate_cuda_for_worker(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia, worker GPU tidak bisa dijalankan.")
    if device == "cuda":
        return
    try:
        gpu_index = int(device.split(":", maxsplit=1)[1])
    except Exception as exc:
        raise ValueError(f"Format device tidak valid: {device}") from exc
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Device {device} tidak tersedia. GPU terdeteksi: {torch.cuda.device_count()}."
        )


def _append_rows_jsonl_locked(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_worker(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation_path)
    output_path = Path(args.output_path)
    image_root = Path(args.image_root)
    cache_dir = Path(args.cache_dir)
    local_model_dir = Path(args.local_model_dir)

    record_limit = _resolve_record_limit(args.sample_limit, args.num_datasets)
    records = _prepare_records(annotation_path, record_limit)
    class_names = _parse_class_names(args.class_names, records)
    emotion_list_text = ", ".join(class_names)
    prompt_with_box = _render_prompt(args.prompt_with_box_template, emotion_list_text)
    prompt_no_box = _render_prompt(args.prompt_no_box_template, emotion_list_text)

    existing_by_id = _load_existing_map(output_path, resume=args.resume)
    existing_ids = set(existing_by_id.keys())

    assigned_total = sum(1 for idx in range(len(records)) if idx % args.num_shards == args.shard_id)
    pending_total = sum(
        1
        for idx, row in enumerate(records)
        if idx % args.num_shards == args.shard_id and _sample_id(row, idx) not in existing_ids
    )

    print(
        f"[worker {args.shard_id}] assigned={assigned_total} pending={pending_total} "
        f"output={output_path}"
    )

    if pending_total == 0:
        print(f"[worker {args.shard_id}] Tidak ada data baru untuk diproses.")
        return

    device = args.device or "cuda"
    _validate_cuda_for_worker(device)

    if not _local_model_ready(local_model_dir):
        raise RuntimeError(
            "Model lokal belum siap. Jalankan mode launcher terlebih dahulu agar model dipersiapkan."
        )

    processor = AutoProcessor.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    model = _load_qwen_model(local_model_dir, device=device, cache_dir=cache_dir)
    print(f"[worker {args.shard_id}] device={device} model_loaded_from_local={local_model_dir}")

    buffered_rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    processed_new = 0

    for index, row in enumerate(tqdm(records, desc=f"worker-{args.shard_id}")):
        if index % args.num_shards != args.shard_id:
            continue

        sid = _sample_id(row, index)
        if sid in existing_ids:
            continue

        try:
            image_ref = Path(str(row["image_path"]))
            image_path = image_ref if image_ref.is_absolute() else image_root / image_ref
            image = load_rgb_image(image_path)

            bbox = row.get("bbox")
            if bbox not in (None, "", "nan"):
                image_for_prompt = draw_red_box(image, bbox)
                prompt_text = prompt_with_box
            else:
                image_for_prompt = image
                prompt_text = prompt_no_box

            generated_text = _generate_caption(
                model=model,
                processor=processor,
                image=image_for_prompt,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
            )

            updated = dict(row)
            updated.setdefault("sample_id", sid)
            updated["semantic_pseudo_label"] = generated_text
            buffered_rows.append(updated)

        except Exception as exc:
            failed = dict(row)
            failed.setdefault("sample_id", sid)
            failed["semantic_pseudo_label"] = ""
            failed["stage1_error"] = str(exc)
            buffered_rows.append(failed)
            errors.append((sid, str(exc)))

        existing_ids.add(sid)
        processed_new += 1

        if len(buffered_rows) >= args.save_every:
            _append_rows_jsonl_locked(output_path, buffered_rows)
            buffered_rows = []

    _append_rows_jsonl_locked(output_path, buffered_rows)

    print(f"[worker {args.shard_id}] done. generated_new={processed_new}")
    if errors:
        print(f"[worker {args.shard_id}] generation_errors={len(errors)} first_3={errors[:3]}")


def run_launcher(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)
    local_model_dir = Path(args.local_model_dir)

    if args.num_shards <= 0:
        raise ValueError("--num-shards harus > 0")
    if args.save_every <= 0:
        raise ValueError("--save-every harus > 0")

    record_limit = _resolve_record_limit(args.sample_limit, args.num_datasets)
    records = _prepare_records(annotation_path, record_limit)
    if not records:
        raise ValueError("Record annotation kosong, tidak ada data untuk diproses.")

    existing_by_id = _load_existing_map(output_path, resume=args.resume)
    completed_ids = set(existing_by_id.keys())
    next_index, next_sid = _first_missing(records, completed_ids)

    print(f"Total records (target run): {len(records)}")
    print(f"Completed from output file: {len(completed_ids)}")
    print(f"Pending records: {len(records) - len(completed_ids)}")
    if next_index is None:
        print("Semua sample target sudah ada di output. Tidak ada proses baru.")
        return

    print(f"Resume mulai dari gambar index: {next_index}")
    print(f"Resume sample_id: {next_sid}")

    if args.check_only:
        print("Check-only aktif. Worker tidak dijalankan.")
        return

    gpu_list = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_list) < args.num_shards:
        raise ValueError(
            f"Jumlah GPU yang diberikan ({len(gpu_list)}) kurang dari num_shards ({args.num_shards})."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia, pipeline dual-GPU tidak bisa dijalankan.")

    gpu_indices: list[int] = []
    for gpu_id in gpu_list[: args.num_shards]:
        try:
            gpu_indices.append(int(gpu_id))
        except Exception as exc:
            raise ValueError(f"GPU id tidak valid: {gpu_id}") from exc

    total_gpu = torch.cuda.device_count()
    for gpu_index in gpu_indices:
        if gpu_index >= total_gpu:
            raise RuntimeError(
                f"GPU id {gpu_index} tidak tersedia. GPU terdeteksi: {total_gpu}."
            )

    _ensure_local_model(args.model_id, local_model_dir=local_model_dir, cache_dir=cache_dir)

    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    worker_processes: list[subprocess.Popen[Any]] = []
    script_path = Path(__file__).resolve()

    for shard_id in range(args.num_shards):
        device = f"cuda:{gpu_indices[shard_id]}"
        cmd = [
            sys.executable,
            str(script_path),
            "--mode",
            "worker",
            "--annotation-path",
            str(annotation_path),
            "--image-root",
            str(args.image_root),
            "--output-path",
            str(output_path),
            "--model-id",
            str(args.model_id),
            "--local-model-dir",
            str(local_model_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--save-every",
            str(args.save_every),
            "--num-shards",
            str(args.num_shards),
            "--shard-id",
            str(shard_id),
            "--device",
            device,
            "--prompt-with-box-template",
            str(args.prompt_with_box_template),
            "--prompt-no-box-template",
            str(args.prompt_no_box_template),
            "--class-names",
            str(args.class_names),
        ]

        if args.num_datasets is not None:
            cmd.extend(["--num-datasets", str(args.num_datasets)])
        elif args.sample_limit is not None:
            cmd.extend(["--sample-limit", str(args.sample_limit)])

        if args.resume:
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")

        print(f"Launching worker {shard_id} on {device}")
        worker_processes.append(subprocess.Popen(cmd))

    failed_workers: list[int] = []
    for shard_id, process in enumerate(worker_processes):
        return_code = process.wait()
        if return_code != 0:
            failed_workers.append(shard_id)

    if failed_workers:
        raise RuntimeError(f"Worker gagal: {failed_workers}")

    merged_by_id = _load_existing_map(output_path, resume=True)
    merged_rows = _ordered_global(records, merged_by_id)
    save_records(output_path, merged_rows)

    merged_ids = {
        str(row.get("sample_id", "")).strip()
        for row in merged_rows
        if str(row.get("sample_id", "")).strip()
    }
    final_next_index, final_next_sid = _first_missing(records, merged_ids)

    print(f"Merged output saved: {output_path}")
    print(f"Merged rows (deduplicated): {len(merged_rows)}")
    if final_next_index is None:
        print("Semua data target sudah tergenerate.")
    else:
        print(f"Masih ada data tersisa. Next missing index: {final_next_index}, sample_id: {final_next_sid}")


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
        return
    run_launcher(args)


if __name__ == "__main__":
    main()
