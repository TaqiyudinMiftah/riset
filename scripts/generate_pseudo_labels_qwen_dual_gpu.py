from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

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

DEFAULT_PROMPT_WITH_BOX_TEMPLATE = (
    "Given the following list of emotions: {emotion_list}, please explain in detail "
    "which emotions are most suitable for describing how the person in the red box feels "
    "based on image context. Focus on facial expressions, body pose, and environmental cues."
)

DEFAULT_PROMPT_NO_BOX_TEMPLATE = (
    "Given the following list of emotions: {emotion_list}, please explain in detail "
    "which emotions are most suitable for describing how the main person feels based on "
    "image context. Focus on facial expressions, body pose, and environmental cues."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qwen pseudo-labels with dual-GPU sharding and resume support.")
    parser.add_argument("--mode", choices=["launcher", "worker"], default="launcher")

    parser.add_argument(
        "--annotation-path",
        default=str(PROJECT_ROOT / "notebook_outputs" / "risetv1_qwen" / "annotations" / "caers_annotations_with_val.jsonl"),
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
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "model_cache"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=None)
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


def _prepare_records(annotation_path: Path, sample_limit: int | None) -> list[dict[str, Any]]:
    records = load_records(annotation_path)
    if sample_limit is not None:
        records = records[: int(sample_limit)]
    return records


def _load_rows_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_records(path)


def _shard_output_path(output_path: Path, shard_id: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.shard{shard_id}{output_path.suffix}")


def _collect_existing_ids(paths: list[Path]) -> set[str]:
    existing: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_records(path):
            sid = str(row.get("sample_id", "")).strip()
            if sid:
                existing.add(sid)
    return existing


def _first_missing(records: list[dict[str, Any]], completed_ids: set[str]) -> tuple[int | None, str | None]:
    for index, row in enumerate(records):
        sid = _sample_id(row, index)
        if sid not in completed_ids:
            return index, sid
    return None, None


def _ordered_for_shard(
    records: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    shard_id: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        if index % num_shards != shard_id:
            continue
        sid = _sample_id(row, index)
        if sid in by_id:
            ordered.append(by_id[sid])
    return ordered


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


def _load_qwen_model(model_id: str, device: str, cache_dir: Path) -> torch.nn.Module:
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "cache_dir": str(cache_dir),
    }

    if Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

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
    try:
        gpu_index = int(device.split(":", maxsplit=1)[1])
    except Exception as exc:
        raise ValueError(f"Format device tidak valid: {device}") from exc
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Device {device} tidak tersedia. GPU terdeteksi: {torch.cuda.device_count()}."
        )


def run_worker(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation_path)
    output_path = Path(args.output_path)
    image_root = Path(args.image_root)
    cache_dir = Path(args.cache_dir)

    records = _prepare_records(annotation_path, args.sample_limit)
    class_names = _parse_class_names(args.class_names, records)
    emotion_list_text = ", ".join(class_names)
    prompt_with_box = args.prompt_with_box_template.format(emotion_list=emotion_list_text)
    prompt_no_box = args.prompt_no_box_template.format(emotion_list=emotion_list_text)

    shard_path = _shard_output_path(output_path, args.shard_id)
    all_shard_paths = [_shard_output_path(output_path, shard) for shard in range(args.num_shards)]

    existing_paths = [output_path, *all_shard_paths] if args.resume else []
    existing_ids = _collect_existing_ids(existing_paths)

    previous_shard_rows = _load_rows_if_exists(shard_path) if args.resume else []
    results_by_id: dict[str, dict[str, Any]] = {}
    for row in previous_shard_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            results_by_id[sid] = row

    assigned_total = sum(1 for idx in range(len(records)) if idx % args.num_shards == args.shard_id)
    pending_total = sum(
        1
        for idx, row in enumerate(records)
        if idx % args.num_shards == args.shard_id and _sample_id(row, idx) not in existing_ids
    )

    print(
        f"[worker {args.shard_id}] assigned={assigned_total} pending={pending_total} "
        f"output={shard_path}"
    )

    if pending_total == 0:
        if results_by_id:
            save_records(shard_path, _ordered_for_shard(records, results_by_id, args.shard_id, args.num_shards))
        print(f"[worker {args.shard_id}] Tidak ada data baru untuk diproses.")
        return

    device = args.device or "cuda"
    _validate_cuda_for_worker(device)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    model = _load_qwen_model(args.model_id, device=device, cache_dir=cache_dir)
    print(f"[worker {args.shard_id}] device={device} model_loaded")

    generated_since_save = 0
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
            results_by_id[sid] = updated

        except Exception as exc:
            failed = dict(row)
            failed.setdefault("sample_id", sid)
            failed["semantic_pseudo_label"] = ""
            failed["stage1_error"] = str(exc)
            results_by_id[sid] = failed
            errors.append((sid, str(exc)))

        existing_ids.add(sid)
        processed_new += 1
        generated_since_save += 1

        if generated_since_save >= args.save_every:
            partial_rows = _ordered_for_shard(records, results_by_id, args.shard_id, args.num_shards)
            save_records(shard_path, partial_rows)
            generated_since_save = 0

    final_rows = _ordered_for_shard(records, results_by_id, args.shard_id, args.num_shards)
    save_records(shard_path, final_rows)

    print(f"[worker {args.shard_id}] done. generated_new={processed_new} total_saved={len(final_rows)}")
    if errors:
        print(f"[worker {args.shard_id}] generation_errors={len(errors)} first_3={errors[:3]}")


def run_launcher(args: argparse.Namespace) -> None:
    annotation_path = Path(args.annotation_path)
    output_path = Path(args.output_path)

    records = _prepare_records(annotation_path, args.sample_limit)
    shard_paths = [_shard_output_path(output_path, shard) for shard in range(args.num_shards)]

    completed_ids = _collect_existing_ids([output_path, *shard_paths]) if args.resume else set()
    next_index, next_sid = _first_missing(records, completed_ids)

    print(f"Total records: {len(records)}")
    print(f"Completed from existing outputs: {len(completed_ids)}")
    print(f"Pending records: {len(records) - len(completed_ids)}")
    if next_index is None:
        print("Semua sample sudah ada di output. Tidak ada proses baru.")
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
    if torch.cuda.device_count() < args.num_shards:
        raise RuntimeError(
            f"GPU terdeteksi {torch.cuda.device_count()}, tetapi num_shards={args.num_shards}."
        )

    worker_processes: list[subprocess.Popen[Any]] = []
    script_path = Path(__file__).resolve()

    for shard_id in range(args.num_shards):
        device = f"cuda:{gpu_list[shard_id]}"
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

        if args.sample_limit is not None:
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

    merged_by_id: dict[str, dict[str, Any]] = {}
    for path in [output_path, *shard_paths]:
        for row in _load_rows_if_exists(path):
            sid = str(row.get("sample_id", "")).strip()
            if sid:
                merged_by_id[sid] = row

    merged_rows = _ordered_global(records, merged_by_id)
    save_records(output_path, merged_rows)

    merged_ids = {str(row.get("sample_id", "")).strip() for row in merged_rows if str(row.get("sample_id", "")).strip()}
    final_next_index, final_next_sid = _first_missing(records, merged_ids)

    print(f"Merged output saved: {output_path}")
    print(f"Merged rows: {len(merged_rows)}")
    if final_next_index is None:
        print("Semua data sudah tergenerate.")
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
