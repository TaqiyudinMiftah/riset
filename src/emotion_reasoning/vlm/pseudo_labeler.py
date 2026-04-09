"""Semantic pseudo-label generation with VLMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

from emotion_reasoning.constants import get_class_names
from emotion_reasoning.utils.image_ops import draw_red_box, load_rgb_image
from emotion_reasoning.utils.io import load_records, save_records

DEFAULT_PROMPT_TEMPLATE = (
    "Given the following list of emotions: {emotion_list}, please explain in detail "
    "which emotions are more suitable for describing how the person in the red box feels "
    "based on the image context. Focus on facial expressions, body pose, and environmental cues."
)

DEFAULT_PROMPT_TEMPLATE_NO_BOX = (
    "Given the following list of emotions: {emotion_list}, please explain in detail "
    "which emotions are more suitable for describing how the main person feels based on "
    "the image context. Focus on facial expressions, body pose, and environmental cues."
)


class VLMAdapter(Protocol):
    def generate(self, image: Image.Image, prompt: str) -> str:
        ...


@dataclass(slots=True)
class VLMGenerationConfig:
    model_name: str
    vlm_type: str
    device: str
    max_new_tokens: int = 192


class LlavaAdapter:
    def __init__(self, config: VLMGenerationConfig):
        dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(config.device)
        self.model.eval()
        self.device = config.device
        self.max_new_tokens = config.max_new_tokens

    def generate(self, image: Image.Image, prompt: str) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        prompt_tokens = inputs.get("input_ids")
        if prompt_tokens is not None:
            generated_ids = generated_ids[:, prompt_tokens.shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


class MoondreamAdapter:
    def __init__(self, config: VLMGenerationConfig):
        dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(config.device)
        self.model.eval()
        self.device = config.device

    def generate(self, image: Image.Image, prompt: str) -> str:
        with torch.inference_mode():
            if hasattr(self.model, "query"):
                response = self.model.query(image, prompt)
            elif hasattr(self.model, "answer_question") and hasattr(self.model, "encode_image"):
                encoded_image = self.model.encode_image(image)
                response = self.model.answer_question(encoded_image, prompt, self.tokenizer)
            else:  # pragma: no cover
                raise RuntimeError("Unsupported Moondream2 API. Please update the adapter for your checkpoint version.")
        if isinstance(response, dict):
            return str(response.get("answer", response.get("text", response))).strip()
        return str(response).strip()


def build_vlm_adapter(config: VLMGenerationConfig) -> VLMAdapter:
    vlm_type = config.vlm_type.lower()
    if vlm_type == "llava":
        return LlavaAdapter(config)
    if vlm_type == "moondream":
        return MoondreamAdapter(config)
    raise ValueError(f"Unsupported VLM type: {config.vlm_type}")


def _resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_prompt(dataset_name: str, prompt_template: str, custom_classes: list[str] | None = None) -> str:
    emotion_list = ", ".join(get_class_names(dataset_name, custom_classes))
    return prompt_template.format(emotion_list=emotion_list)


def generate_pseudo_labels(
    annotation_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    dataset_name: str,
    vlm_type: str,
    vlm_model: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    custom_classes: list[str] | None = None,
    max_new_tokens: int = 192,
    device: str | None = None,
    image_column: str = "image_path",
    bbox_column: str = "bbox",
    sample_id_column: str = "sample_id",
    output_column: str = "semantic_pseudo_label",
    no_box_prompt_template: str = DEFAULT_PROMPT_TEMPLATE_NO_BOX
) -> list[dict[str, Any]]:
    records = load_records(annotation_path)
    image_root_path = Path(image_root)
    prompt_with_box = _build_prompt(
        dataset_name=dataset_name,
        prompt_template=prompt_template,
        custom_classes=custom_classes
    )
    prompt_without_box = _build_prompt(
        dataset_name=dataset_name,
        prompt_template=no_box_prompt_template,
        custom_classes=custom_classes
    )
    adapter = build_vlm_adapter(
        VLMGenerationConfig(
            model_name=vlm_model,
            vlm_type=vlm_type,
            device=_resolve_device(device),
            max_new_tokens=max_new_tokens
        )
    )

    enriched_records: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(records, desc="Generating pseudo-labels")):
        image_ref = Path(str(record[image_column]))
        image_path = image_ref if image_ref.is_absolute() else image_root_path / image_ref
        image = load_rgb_image(image_path)
        bbox = record.get(bbox_column)
        prompt = prompt_without_box
        if bbox not in (None, "", "nan"):
            image = draw_red_box(image, bbox)
            prompt = prompt_with_box
        description = adapter.generate(image=image, prompt=prompt)
        updated = dict(record)
        updated.setdefault(sample_id_column, f"sample_{index:06d}")
        updated[output_column] = description
        enriched_records.append(updated)

    save_records(output_path, enriched_records)
    return enriched_records
