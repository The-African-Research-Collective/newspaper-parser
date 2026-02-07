import torch
import numpy as np
import json
import os
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset, load_from_disk, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor
from torch.nn.utils.rnn import pad_sequence

from karanta.training.utils import load_yaml_config, SingleDatapoint
from karanta.training.pipeline_steps import (
    PDF2ImageStep,
    JSONOutputFormat,
    FetchPageData,
    StaticLengthDocumentAnchoring,
    FinetuningPrompt,
    InstructUserMessages,
    Tokenizer,
    FetchMultipageData,
)

str2PipelineStep = {
    "PDF2ImageStep": PDF2ImageStep,
    "JSONOutputFormat": JSONOutputFormat,
    "FetchPageData": FetchPageData,
    "StaticLengthDocumentAnchoring": StaticLengthDocumentAnchoring,
    "FinetuningPrompt": FinetuningPrompt,
    "InstructUserMessages": InstructUserMessages,
    "Tokenizer": Tokenizer,
    "FetchMultipageData": FetchMultipageData,
}


def check_tokens_and_labels(input_ids, labels):
    # Count total tokens in input_ids
    total_tokens = input_ids.numel()

    # Find all non -100 items in labels
    non_padding_mask = labels != -100
    non_padding_items = labels[non_padding_mask]
    valid_label_count = non_padding_items.numel()

    # Print results
    print(f"Total tokens in input_ids: {total_tokens}")
    print(f"Valid labels (non -100): {valid_label_count}")
    print(
        f"Percentage of valid labels: {(valid_label_count / total_tokens * 100):.2f}%"
    )

    return total_tokens, valid_label_count, non_padding_items


def initialize_dataset(
    json_dir: Path, pdf_dir: Path, max_workers: Optional[int] = None
):
    json_files = list(json_dir.glob("*.json"))
    random.shuffle(json_files)

    def check_pair(json_file):
        pdf_file = pdf_dir / f"{json_file.stem}.pdf"
        if pdf_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    _ = json.loads(f.read())["generation"]["pages"]
                return {"pdf_path": str(pdf_file), "json_path": str(json_file)}
            except Exception:
                return None
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(filter(None, ex.map(check_pair, json_files[:100000])))
    return results


class LocalDataset:
    """
    Reimplementation of LocalDataset with Hugging Face caching, preserving sequential pipeline dependencies.
    Each sample runs the full pipeline (step-by-step), and the final model_inputs are cached as Arrow.
    """

    def __init__(
        self,
        root_dir: Path,
        pdf_dir_name: str = "pdf_inputs",
        json_dir_name: str = "json_outputs",
        cache_folder_name: Optional[str] = None,
        pipeline_steps: Optional[List[Dict]] = None,
        num_samples: int = -1,
        force_rebuild: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.pdf_dir = self.root_dir / pdf_dir_name
        self.json_dir = self.root_dir / json_dir_name

        # === cache dir ===
        cache_folder_name = cache_folder_name or "processed_hf_seq"
        self.cache_dir = Path(self.root_dir) / ".cache" / cache_folder_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # === pipeline fingerprint ===
        pipeline_hash = hashlib.md5(
            json.dumps(pipeline_steps or [], sort_keys=True).encode()
        ).hexdigest()
        self.cache_path = self.cache_dir / f"dataset_{pipeline_hash}"

        print(f"🔍 Dataset cache path: {self.cache_path}")
        self.temp_cache_paths = list(self.cache_dir.glob("dataset_*"))
        if self.temp_cache_paths and not force_rebuild:
            print(
                f"⚠️ Warning: Found existing cache folders in {self.cache_dir}, they may be outdated: {self.temp_cache_paths}"
            )
            for i in range(len(self.temp_cache_paths)):
                try:
                    print(
                        f"📦 Loading cached dataset from {self.temp_cache_paths[i]}..."
                    )
                    self.dataset = load_from_disk(str(self.temp_cache_paths[i]))
                    self.dataset.select(range(num_samples))
                    return
                except Exception:
                    print("❌ Failed to load existing cache")
                    continue

        print("🧩 Building dataset with sequential pipeline execution...")
        raw_samples = initialize_dataset(self.json_dir, self.pdf_dir)
        if num_samples > 0:
            raw_samples = raw_samples[:num_samples]
        self.raw_dataset = Dataset.from_list(raw_samples)

        # === Initialize pipeline ===
        self.pipeline = self._initialize_pipeline_steps(pipeline_steps)

        # === Apply full pipeline per example ===
        def full_pipeline(examples):
            input_ids_list = []
            attention_masks_list = []
            labels_list = []
            pixel_values_list = []
            image_grid_thw_list = []

            examples_zipped = zip(examples["pdf_path"], examples["json_path"])

            for example in examples_zipped:
                sample = SingleDatapoint(
                    pdf_path=Path(example[0]), json_path=Path(example[1])
                )
                try:
                    for step in self.pipeline:
                        sample = step(sample)

                    if sample.model_inputs is None:
                        continue

                    input_ids_list.append(sample.model_inputs["input_ids"])
                    attention_masks_list.append(sample.model_inputs["attention_mask"])
                    labels_list.append(sample.model_inputs["labels"])
                    pixel_values_list.append(sample.model_inputs["pixel_values"])
                    image_grid_thw_list.append(sample.model_inputs["image_grid_thw"])
                except Exception as e:
                    print(e)
                    continue

            examples["input_ids"] = input_ids_list
            examples["attention_mask"] = attention_masks_list
            examples["labels"] = labels_list
            examples["pixel_values"] = pixel_values_list
            examples["image_grid_thw"] = image_grid_thw_list

            return examples

        # === Split and process in chunks of 50k ===
        chunk_size = 50_000
        total_samples = len(self.raw_dataset)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        processed_datasets = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            print(f"🚀 Processing chunk {i + 1}/{num_chunks}: samples {start}–{end}")

            subset = self.raw_dataset.select(range(start, end))
            processed_chunk = subset.map(
                full_pipeline,
                desc=f"Applying full Karanta pipeline [chunk {i+1}]",
                num_proc=48,
                remove_columns=self.raw_dataset.column_names,
                batched=True,
                batch_size=4,
                cache_file_name=str(self.cache_dir / f"hf_cache_chunk_{i+1}.arrow"),
            )

            processed_datasets.append(processed_chunk)

            if i == 1:
                break

        # === Concatenate all processed chunks ===
        print("🧩 Concatenating all processed chunks...")
        self.dataset = concatenate_datasets(processed_datasets)

        # === Persist Arrow dataset ===
        self.dataset.save_to_disk(str(self.cache_path),num_proc=32)

    def _initialize_pipeline_steps(self, pipeline_config: List[Dict]):
        steps = []
        for cfg in pipeline_config:
            cfg = cfg.copy()
            name = cfg.pop("name")
            step_class = str2PipelineStep[name]
            if name == "Tokenizer":
                processor = AutoProcessor.from_pretrained(cfg.pop("processor"))
                steps.append(step_class(**cfg, processor=processor))
            else:
                steps.append(step_class(**cfg))
        return steps

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class DataCollator:
    """
    Prepares a batch of multimodal OCR samples for training Qwen2.5-VL-3B-Instruct.
    Handles tokenizer-based padding for text and dynamic padding for image embeddings.
    """

    def __init__(self, model_name_or_path: str, max_token_len: Optional[int] = None):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.max_token_len = max_token_len
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.masking_index = -100  # For label ignore

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, attention_masks_list, labels_list = [], [], []
        pixel_values_list = []
        image_grid_thw = []

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if isinstance(x, list):
                return torch.tensor(x)
            return x

        for sample in batch:
            if not sample:
                continue

            input_ids = to_tensor(sample["input_ids"]).squeeze(0)
            attention_mask = to_tensor(sample["attention_mask"]).squeeze(0)
            labels = to_tensor(sample["labels"]).squeeze(0)

            if self.max_token_len is not None:
                input_ids = input_ids[: self.max_token_len]
                attention_mask = attention_mask[: self.max_token_len]
                labels = labels[: self.max_token_len]

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            labels_list.append(labels)

            # pixel_values may be [num_patches, hidden_dim] → variable num_patches
            pixel_values = to_tensor(sample["pixel_values"])
            pixel_values_list.append(pixel_values)

            image_grid_thw_values = to_tensor(sample["image_grid_thw"])
            image_grid_thw.append(image_grid_thw_values)

        if not input_ids_list:
            return {}

        # === Pad text using tokenizer ===
        batch_encodings = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_masks_list,
            },
            padding=True,
            return_tensors="pt",
        )

        labels_padded = self.processor.tokenizer.pad(
            {"input_ids": labels_list},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels_padded[labels_padded == self.pad_token_id] = self.masking_index

        # === Pad visual embeddings ===
        # pixel_values: list of [num_patches, hidden_dim] tensors → pad along num_patches
        pixel_values_padded = pad_sequence(
            pixel_values_list, batch_first=True, padding_value=0.0
        )

        batch_dict = {
            "input_ids": batch_encodings["input_ids"],
            "attention_mask": batch_encodings["attention_mask"],
            "labels": labels_padded,
            "pixel_values": pixel_values_padded,
            "image_grid_thw": torch.stack(image_grid_thw),
        }

        return batch_dict


if __name__ == "__main__":
    all_config = load_yaml_config(
        "configs/training/ocr/karanta_set_qwen_3_4B_vl_all_linear_no_base_text.yaml"
    )
    # print(all_config)
    config = all_config["dataset_train"][0]
    pipeline = config["pipeline"]

    print(pipeline)
    dataset = LocalDataset(
        root_dir=Path(config["root_dir"]),
        pdf_dir_name=config["pdf_dir_name"],
        json_dir_name=config["json_dir_name"],
        cache_folder_name=all_config.get("data_cache_folder_name", None),
        num_samples=-1,
        pipeline_steps=pipeline,
    )

    # ts = concatenate_datasets([dataset.dataset, dataset.dataset])
    ts = dataset.dataset

    print(dataset)
    print(dataset[0].keys())
    print(len(ts))
    # print(dataset[0]['input_ids'].shape)

    # from transformers import AutoProcessor

    # # torch.set_printoptions(threshold=10_000)
    # # numpy.set_printoptions(threshold=10_000)

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # print(processor.tokenizer.padding_side)

    # print(
    #     processor(text = ["I am the prince of wales"], return_tensors="np",padding="max_length", max_length=200, )
    # )
    # for i in range(len(dataset)):
    #     print(dataset[i]['input_ids'])
    #     print(dataset[i]['input_ids'].shape)
    # print(processor.tokenizer.pad_token_id)
    # print(processor.tokenizer.padding_side)

    # print(processor.tokenizer.decode(dataset[0]['input_ids']))

    # print(processor.tokenizer.pad_token_id in dataset[0]['input_ids'])

    # print(f"Dataset length: {len(dataset)}")

    # dataloader = DataLoader(
    #     ts,
    #     batch_size=0,
    #     collate_fn=DataCollator(
    #         "Qwen/Qwen2.5-VL-3B-Instruct", max_token_len=all_config["max_length"]
    #     ),
    #     num_workers=4,
    # )
    # from tqdm import tqdm

    # for b in tqdm(dataloader):
    #     print(b.keys())
    #     break

    # print(next(iter(dataloader))['input_ids'].shape)

    # from tqdm import tqdm

    # for sample in tqdm(iter(dataloader), total=len(dataset)):

    #     print(sample['input_ids'].shape)
    #     print(sample['attention_mask'].shape)
    #     print(sample['labels'].shape)
    #     print(torch.sum(sample['attention_mask']))

    #     check_tokens_and_labels(sample['input_ids'], sample['labels'])
    #     print("====================================")

    # print(f"Dataset samples: {dataset[0].user_messages}")

    # from transformers import AutoProcessor

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # text = processor.apply_chat_template(
    #     [dataset[0].user_messages],  # Wrap in list as expected by the template
    #     tokenize=False,  # Keep as text, don't tokenize yet
    #     add_generation_prompt=True,  # Add generation tokens depending on the model
    # )
    # print(text)
