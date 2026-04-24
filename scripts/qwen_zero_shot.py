from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    parser = argparse.ArgumentParser(
        description="Run zero-shot semantic label inference with Qwen and keep a local model copy."
    )
    parser.add_argument(
        "--annotation-path",
        default=str(
            PROJECT_ROOT
            / "notebook_outputs"
            / "risetv1_qwen"
            / "annotations"
            / "caers_annotations_with_val.jsonl"
        ),
    )
    parser.add_argument(
        "--image-root",
        default=str(PROJECT_ROOT / "caer_dataset" / "CAER-S"),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT
            / "notebook_outputs"
            / "risetv1_qwen"
            / "stage1_semantic_zero_shot_qwen.jsonl"
        ),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--local-model-dir",
        default=str(PROJECT_ROOT / "model_cache" / "qwen3_vl_4b_instruct"),
    )
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "model_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--class-names", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--check-model-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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


def _load_existing_map(output_path: Path, resume: bool) -> dict[str, dict[str, Any]]:
    existing_by_id: dict[str, dict[str, Any]] = {}
    if not resume or not output_path.exists():
        return existing_by_id

    for row in load_records(output_path):
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


def _ordered_results(records: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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

    labels: set[str] = set()
    for row in records:
        value = row.get("labels", "")
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item).strip()
                if text:
                    labels.add(text)
            continue

        text = str(value).strip()
        if text:
            labels.add(text)

    return sorted(labels)


def _resolve_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _validate_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia, tidak bisa menjalankan mode GPU.")
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


def _local_model_ready(local_model_dir: Path) -> bool:
    if not local_model_dir.exists():
        return False

    manifest_path = local_model_dir / "model_manifest.json"
    if not manifest_path.exists():
        return False

    has_config = (local_model_dir / "config.json").exists()
    has_weights = any(local_model_dir.glob("*.safetensors")) or any(local_model_dir.glob("pytorch_model*"))
    return has_config and has_weights


def _load_qwen_components(
    source: str | Path,
    device: str,
    cache_dir: Path,
    local_only: bool,
) -> tuple[AutoProcessor, torch.nn.Module]:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(device),
        "low_cpu_mem_usage": True,
        "cache_dir": str(cache_dir),
        "local_files_only": local_only,
    }

    processor = AutoProcessor.from_pretrained(
        source,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        local_files_only=local_only,
    )

    if Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(source, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(source, **model_kwargs)

    model = model.to(device)
    model.eval()
    return processor, model


def _write_model_manifest(local_model_dir: Path, model_id: str, source: str) -> None:
    manifest = {
        "model_id": model_id,
        "source": source,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (local_model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_model(
    model_id: str,
    local_model_dir: Path,
    cache_dir: Path,
    device: str,
) -> tuple[AutoProcessor, torch.nn.Module, str]:
    local_model_dir.mkdir(parents=True, exist_ok=True)

    if _local_model_ready(local_model_dir):
        print(f"Model lokal sudah tersedia: {local_model_dir}")
        processor, model = _load_qwen_components(local_model_dir, device=device, cache_dir=cache_dir, local_only=True)
        return processor, model, "local"

    try:
        processor, model = _load_qwen_components(model_id, device=device, cache_dir=cache_dir, local_only=True)
        source = "hf_cache"
        print(f"Model ditemukan di cache lokal Hugging Face untuk {model_id}")
    except Exception:
        print(f"Model belum ada di cache lokal. Download dari Hugging Face: {model_id}")
        processor, model = _load_qwen_components(model_id, device=device, cache_dir=cache_dir, local_only=False)
        source = "downloaded"

    print(f"Menyimpan salinan model ke: {local_model_dir}")
    model.save_pretrained(local_model_dir, safe_serialization=True)
    processor.save_pretrained(local_model_dir)
    _write_model_manifest(local_model_dir, model_id=model_id, source=source)
    return processor, model, source


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
        fallback_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
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


def main() -> None:
    args = parse_args()

    annotation_path = Path(args.annotation_path)
    output_path = Path(args.output_path)
    image_root = Path(args.image_root)
    cache_dir = Path(args.cache_dir)
    local_model_dir = Path(args.local_model_dir)

    records = _prepare_records(annotation_path, args.sample_limit)
    if not records:
        raise ValueError("Record annotation kosong, tidak ada data untuk diproses.")

    class_names = _parse_class_names(args.class_names, records)
    emotion_list_text = ", ".join(class_names)
    prompt_with_box = args.prompt_with_box_template.format(emotion_list=emotion_list_text)
    prompt_no_box = args.prompt_no_box_template.format(emotion_list=emotion_list_text)

    existing_by_id = _load_existing_map(output_path, resume=args.resume)
    completed_ids = set(existing_by_id.keys())
    next_index, next_sid = _first_missing(records, completed_ids)

    print(f"Total records: {len(records)}")
    print(f"Completed from existing output: {len(completed_ids)}")
    print(f"Pending records: {len(records) - len(completed_ids)}")
    if next_index is None:
        print("Semua sample sudah ada di output. Tidak ada proses baru.")
        return

    print(f"Resume mulai dari gambar index: {next_index}")
    print(f"Resume sample_id: {next_sid}")

    _validate_cuda_device(args.device)
    processor, model, model_source = _prepare_model(
        model_id=args.model_id,
        local_model_dir=local_model_dir,
        cache_dir=cache_dir,
        device=args.device,
    )
    print(f"Model siap digunakan dari source: {model_source}")

    if args.check_model_only:
        print("Check-model-only aktif. Inference tidak dijalankan.")
        return

    generated_since_save = 0
    errors: list[tuple[str, str]] = []
    processed_new = 0

    for index, row in enumerate(tqdm(records, desc="zero-shot-inference")):
        sid = _sample_id(row, index)
        if sid in completed_ids:
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
            existing_by_id[sid] = updated

        except Exception as exc:
            failed = dict(row)
            failed.setdefault("sample_id", sid)
            failed["semantic_pseudo_label"] = ""
            failed["stage1_error"] = str(exc)
            existing_by_id[sid] = failed
            errors.append((sid, str(exc)))

        completed_ids.add(sid)
        processed_new += 1
        generated_since_save += 1

        if generated_since_save >= args.save_every:
            save_records(output_path, _ordered_results(records, existing_by_id))
            generated_since_save = 0

    save_records(output_path, _ordered_results(records, existing_by_id))

    print(f"Done. generated_new={processed_new} output={output_path}")
    if errors:
        print(f"generation_errors={len(errors)} first_3={errors[:3]}")


if __name__ == "__main__":
    main()