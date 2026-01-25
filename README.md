<div align="center">
<img src="assets/karanta.png" alt="Karanta OCR Logo" width="300"/>
<br/>
  <br>
  <h1>Karanta OCR</h1>
</div>

# KarantaOCR: Efficient Document Processing for African Languages
Karanta means "read" in Hausa, a language spoken in Nigeria and other West African countries. This project is a OCR toolkit for processing scanned documents containing content in african languages and extracting the text in them at scale.

We would like to give huge credits to the [OlmoOCR](https://github.com/allenai/olmocr) project and team for providing the blueprint that we've adapted for Karanta.

## Table of Contents

- [Model Description](#model-description)
- [Training Data](#training-data)
  - [Stage 1: General OCR Training](#stage-1-general-ocr-training)
  - [Stage 2: African Language Fine-Tuning](#stage-2-african-language-fine-tuning)
  - [Training Plots](#training-plots)
- [Capabilities](#capabilities)
- [Evaluation](#evaluation)
  - [Results -- KarantaOCR-Bench](#results----karantaocr-bench)
  - [Results -- OlmoOCR-Bench](#results----olmocr-bench)
- [How to Use](#how-to-use)
  - [Load the Model and Processor](#load-the-model-and-processor)
  - [Prepare a PDF Page for Inference](#prepare-a-pdf-page-for-inference)
  - [Run OCR Inference](#run-ocr-inference)
  - [End-to-End Example](#end-to-end-example)
- [Citation Information](#citation-information)

---

## Model Description

#### [Paper](....)

**KarantaOCR** is an open-source document OCR and processing model designed for **high-accuracy text extraction in African languages**.
The model focuses on preserving language-specific characters and diacritics that are often lost, normalized, or mis-transcribed by existing OCR systems.

KarantaOCR is fine-tuned from [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), a vision-language model that combines a strong vision encoder with a large language model.
Through targeted curriculum fine-tuning, KarantaOCR extends these capabilities to robust document understanding across diverse PDF formats and multilingual settings.

## Training Data

KarantaOCR was trained using a **two-stage curriculum fine-tuning strategy**.

### Stage 1: General OCR Training

* **100,000 documents** sampled from [Allenai OCRMix](allenai/olmOCR-mix-0225)
* Purpose: learn general OCR skills across layouts, fonts, tables, and document structures

### Stage 2: African Language Fine-Tuning

* **50,000 PDFs** containing text in **10 African languages**, crawled from the web
* Domains include:

  * Religious texts
  * Legal documents
  * Dictionaries
  * Novels
  * Other long-form and structured documents

This stage emphasizes accurate transcription of **diacritics, special characters, and region-specific typography**.

### Training Plots

<div align="left">
<img src="https://cdn-uploads.huggingface.co/production/uploads/604b97e27032db3f5e8d6e8e/5Jf26jOGs12rrMwy3hwrI.png" alt="Train Loss" width="600"/>
</div>

<div align="left">
<img src="https://cdn-uploads.huggingface.co/production/uploads/604b97e27032db3f5e8d6e8e/9z4t6so8DIaykrFQHs0Su.png" alt="Eval Loss" width="600"/>
</div>

<div align="left">
<img src="https://cdn-uploads.huggingface.co/production/uploads/604b97e27032db3f5e8d6e8e/-TJltGBXNFABTkShCyvsL.png" alt="Learning Rate" width="600"/>
</div>


---

## Capabilities

KarantaOCR supports:

* High-accuracy **text extraction** from PDFs
* **Table extraction** and structured document understanding
* Robust handling of:

  * Multi-column layouts
  * Headers and footers
  * Mixed scanned and digital PDFs

While improved performance on African languages was our priority, KarantaOCR **maintains strong performance on English and other high-resource languages**, making it suitable for mixed-language document collections.

## Evaluation

KarantaOCR is evaluated on the OLMOocr benchmark using pass-rate accuracy. Scores are reported as averages across JSONL files with 95% confidence intervals.
In addition to OLMOocr benchmark, we also create a KarantaOCR-Bench, which focuses specifically on testing OCR extraction on special characters and diacritics.

### Results -- KarantaOCR-Bench

| JSONL File      | KarantaOCR (3B) | RoLMOCR (7B) | NanoNetsOCR-2 (3B)  | OLMOCR-2 (7B)  |
| --------------- | ---------- | -------- | ------------- | -------- | 
| Diacritics & Special Characters | **61.6** | 53.6 | 55.1 | 55.5

### Results -- OlmoOCR-Bench

| JSONL File      | KarantaOCR (3B) | RoLMOCR (7B) | NanoNetsOCR-2 (3B)  | OLMOCR-1 (7B)  |  OLMOCR-2 (7B)  | Mistral OCR API | DeepSeek-OCR (3B) |
| --------------- | ---------- | -------- | ------------- | -------- | -------- |-------- |-------- |
| arxiv_math      | 74.2       | 76.8 | 73.7          | 63.3     | 83.0 | 77.2 | 77.2 |
| baseline        | 99.4  | 97.9     | 99.5     | 97.9     | 99.7 | 99.4 | 99.8 |
| headers_footers | 95.3   | 94.1     | 32.8          | 93.4     | 96.1 | 93.6 | 96.1 |
| long_tiny_text  | 72.2       | 61.3     | 92.1     | 54.8     | 81.9 | 77.1 | 79.4 |
| multi_column    | 75.6       | 70.0     | 82.5      | 67.6     | 83.7 | 71.3 | 66.4 |
| old_scans       | 41.3       | 42.4     | 41.4          | 38.6     | 47.7 | 29.3 | 33.3 |
| old_scans_math  | 70.3       | 80.1 | 44.1          | 67.5     | 82.3 | 67.5 | 73.6 |
| table_tests     | 64.3       | 72.2     | 84.2      | 62.3     | 84.9 | 60.6 | 80.2 |
| Average         | 74.1%      | 74.4%    | 68.8%         | 68.2%    | 82.4% | 72.0 | 75.7 |


## How to Use  - Inference

KarantaOCR processes PDF documents by rendering pages into images and combining them with structured prompts for inference.

### Load the Model and Processor

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model(model_path: str, device_map: str = "auto", dtype: str = "auto"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype) if dtype != "auto" else "auto",
        device_map=device_map,
    )
    return model

def load_processor(processor_name: str, min_pixels=None, max_pixels=None):
    if min_pixels and max_pixels:
        return AutoProcessor.from_pretrained(
            processor_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
    return AutoProcessor.from_pretrained(processor_name)
```

### Prepare a PDF Page for Inference

```python
from jinja2 import Template

def render_pdf_to_base64png(
    local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048
) -> str:
    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(
                target_longest_image_dim * 72 / longest_dim
            ),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")

def build_message(image_url: str, system_prompt: str, page: int = 0):
    image_base64 = render_pdf_to_base64png(image_url, page, TARGET_IMAGE_DIM)

    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{image_base64}",
                },
            ],
        }
    ]
    return prompt
```

### Run OCR Inference

```python
from qwen_vl_utils import process_vision_info

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

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    outputs = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs[0]
```

### End-to-End Example

```python
model = load_model("taresco/KarantaOCR")
processor = load_processor("taresco/KarantaOCR")

prompt = """Below is the image of one page of a PDF document.
Just return the plain text representation of this document as if you were reading it naturally.
Turn equations into a LaTeX representation, and tables into markdown format. Remove the headers and footers, but keep references and footnotes.
Read any natural handwriting.
This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.
If there is no text at all that you think you should read, you can output null.
if the document contains diacritics, please include them in the output.
Do not hallucinate.
"""

messages = build_message(
    image_url="example.pdf",
    system_prompt=prompt,
    page=0
)

output_text = run_inference(model, processor, messages)
print(output_text)
```

## Setup

Our repository uses uv environment to manage dependencies. To set up the environment, follow these steps:

- Install uv by [following the instrictions here](https://docs.astral.sh/uv/getting-started/installation/)
- Create a new uv environment and install the dependencies:
  ```bash
  uv venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  uv sync
  ```


## Citation Information

Coming soon ...


## License
...


