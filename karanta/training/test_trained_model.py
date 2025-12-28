#!/usr/bin/env python3
import argparse
import yaml
import torch
from jinja2 import Template
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from karanta.data.process_pdf_utils import render_pdf_to_base64png

from karanta.constants import TARGET_IMAGE_DIM


def load_model(model_path: str, device_map: str = "auto", dtype: str = "auto"):
    print(f"Loading model from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype) if dtype != "auto" else "auto",
        device_map=device_map,
    )
    return model


def load_processor(processor_name: str, min_pixels=None, max_pixels=None):
    print(f"Loading processor from {processor_name} ...")
    if min_pixels and max_pixels:
        return AutoProcessor.from_pretrained(
            processor_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
    return AutoProcessor.from_pretrained(processor_name)


def load_prompt_from_yaml(yaml_path: str, key: str) -> str:
    """Load a system prompt string from YAML file using the specified key."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if key not in data:
        raise KeyError(
            f"Prompt key '{key}' not found in {yaml_path}. Available keys: {list(data.keys())}"
        )
    return data[key]


def build_message(image_url: str, system_prompt: str, page: int = 0):
    """Format messages in Qwen VL chat format, injecting system prompt and base text."""

    image_base64 = render_pdf_to_base64png(image_url, page, TARGET_IMAGE_DIM)
    prompt_template_dict = Template(system_prompt)

    print(f"Prompt: {prompt_template_dict.render()}\n=======")

    prompt = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template_dict.render(),
                    },
                    {
                        "type": "image",
                        "image": f"data:image/png;base64,{image_base64}",
                    },
                ],
            }
        ]
    ]

    return prompt[0]


def run_inference(model, processor, messages, max_new_tokens=128, device="cuda"):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=False,
        return_tensors="pt",
    ).to(device)

    print(f"Input token shape: {inputs['input_ids'].shape}")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return outputs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a trained Qwen2.5-VL model with an image and prompt YAML config."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained or base model"
    )
    parser.add_argument(
        "--processor_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Image URL or path to test"
    )
    parser.add_argument(
        "--prompt_yaml",
        type=str,
        required=True,
        help="Path to YAML file containing prompt configs",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        required=True,
        help="Key inside YAML file to use (e.g. system, olmo_ocr_system_prompt)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )

    args = parser.parse_args()

    # Load resources
    model = load_model(args.model_path, dtype=args.dtype)
    processor = load_processor(args.processor_name)
    system_prompt = load_prompt_from_yaml(args.prompt_yaml, args.prompt_key)

    # Build messages
    messages = build_message(args.image, system_prompt)

    # Run inference
    output = run_inference(
        model,
        processor,
        messages,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print("\n=== Model Output ===")
    print(output)


if __name__ == "__main__":
    main()
