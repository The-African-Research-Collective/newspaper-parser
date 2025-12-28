#!/usr/bin/env python3
import os
import json
import argparse
import logging
from math import prod
from pathlib import Path

import torch
from tqdm import tqdm
from tokenizers import models
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from karanta.training.utils import load_yaml_config
from karanta.training.data import LocalDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/vocabtrimmer"


def pretty(num: int) -> str:
    return "{:,}".format(int(num))


def load_model(model_path: str, device_map: str = "auto", dtype: str = "bf16"):
    logger.info(f"Loading model from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype) if dtype != "auto" else "auto",
        device_map=device_map,
    )
    return model


def load_processor(processor_name: str, min_pixels=None, max_pixels=None):
    logger.info(f"Loading processor from {processor_name} ...")
    if min_pixels is not None and max_pixels is not None:
        return AutoProcessor.from_pretrained(
            processor_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
    return AutoProcessor.from_pretrained(processor_name)


def show_parameter(target_model):
    emb = target_model.language_model.get_input_embeddings().weight
    param_size_embedding = prod(emb.shape)

    param_size_full = sum(p.numel() for p in target_model.language_model.parameters())
    vocab_size = emb.shape[0]

    print("PARAMETER SUMMARY")
    print(f"\t*parameter size (full) : {pretty(param_size_full)}")
    print(f"\t*parameter size (vocab): {pretty(param_size_embedding)}")
    print(
        f"\t*parameter size (rest) : {pretty(param_size_full - param_size_embedding)}"
    )
    print(
        f"\t*ratio of vocab param  : {round(param_size_embedding / param_size_full * 100, 1)}%"
    )
    print(f"\t*vocab size            : {pretty(vocab_size)}")
    return param_size_full, param_size_embedding, vocab_size


def build_train_dataset(dataset_config_path: str):
    """
    Build & concatenate all LocalDataset mixes in the YAML config.
    Expects all underlying datasets to contain a 'labels' column (token ids).
    """
    all_config = load_yaml_config(dataset_config_path)

    train_slices = []
    for i, data_mix in enumerate(all_config["dataset_train"]):
        logger.info(
            f"Creating training dataset {i + 1} from: {data_mix.get('root_dir', None)}"
        )
        pipeline_mix = data_mix.get("pipeline", None)

        ld = LocalDataset(
            root_dir=Path(data_mix["root_dir"]),
            pdf_dir_name=data_mix["pdf_dir_name"],
            json_dir_name=data_mix["json_dir_name"],
            pipeline_steps=pipeline_mix,
            cache_folder_name=all_config.get("data_cache_folder_name", None),
            num_samples=-1,
        )

        logger.info(f"Found {len(ld)} samples")
        if len(ld) > 0:
            train_slices.append(ld.dataset)

    if not train_slices:
        raise RuntimeError("No training samples found across dataset_train mixes.")

    train_dataset = (
        concatenate_datasets(train_slices) if len(train_slices) > 1 else train_slices[0]
    )
    logger.info(f"Total training samples: {len(train_dataset)}")
    return train_dataset


def mine_token_frequencies(
    train_dataset,
    vocab_size: int,
    batch_size: int = 64,
    num_workers: int = 8,
    ignore_index: int = -100,
):
    """
    Iterate over dataset with a DataLoader and accumulate token counts via torch.bincount.

    Returns:
      counts: torch.IntTensor [vocab_size] with frequencies.
    """
    # Avoid Arrow serialization issues by NOT using dataset.map for reductions.
    # Using HF's torch formatting helps when possible, but we still robustly handle lists.
    ds = train_dataset
    try:
        ds = ds.with_format("torch", columns=["labels", "input_ids"])
    except Exception as e:
        logger.warning(
            f"with_format('torch') failed; will fall back to manual tensor conversion. Reason: {e}"
        )

    def collate(examples):
        # Keep raw examples; we will flatten labels manually (ragged-safe).
        return examples

    dl = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate
    )

    counts = torch.zeros(int(vocab_size), dtype=torch.int64)

    total_batches = (len(ds) + batch_size - 1) // batch_size
    for batch in tqdm(dl, total=total_batches, desc="Mining vocab freqs"):
        # Flatten token ids across the batch
        flat = []
        for ex in batch:
            lab = ex["labels"]
            input_ids = ex["input_ids"]
            if isinstance(lab, torch.Tensor):
                # combine all the input ids and labels into one tensor
                t = torch.cat(
                    (
                        lab.reshape(-1).to(dtype=torch.long),
                        input_ids.reshape(-1).to(dtype=torch.long),
                    )
                )
            else:
                # list/np array
                t = torch.cat(
                    (
                        torch.tensor(lab, dtype=torch.long).reshape(-1),
                        torch.tensor(input_ids, dtype=torch.long).reshape(-1),
                    )
                )
            flat.append(t)

        # combine all the input ids and labels for all the examples in the batch
        labels = torch.cat(flat, dim=0)

        if ignore_index is not None:
            labels = labels[labels != ignore_index]

        # Safety: keep only valid ids
        labels = labels[(labels >= 0) & (labels < vocab_size)]
        if labels.numel() == 0:
            continue

        counts += torch.bincount(labels, minlength=vocab_size)

    return counts


def save_frequency_json(counts: torch.Tensor, out_path: str):
    """
    Save counts as JSON mapping token_id(str) -> frequency(int)
    Only stores nonzero counts to keep file small.
    """
    freq = {str(i): int(c) for i, c in enumerate(counts.tolist()) if c > 0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(freq, f)
    logger.info(
        f"Saved frequency file: {out_path} (nonzero tokens: {pretty(len(freq))})"
    )


def load_frequency_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_backend_model_state(tokenizer):
    """
    Returns (model_type_str, vocab_as_list_of_pairs, merges_list_or_none, extra_fields_dict)
    for tokenizers backend models.
    """
    state = json.loads(tokenizer.backend_tokenizer.model.__getstate__())
    model_type = state.get("type", None)  # tokenizer type
    if model_type is None:
        raise ValueError(
            f"Tokenizer backend model state missing 'type': keys={list(state.keys())}"
        )

    vocab = state.get("vocab", None)
    merges = state.get("merges", None)

    # Normalize vocab to list[(token, id)]
    if isinstance(vocab, dict):
        vocab_pairs = list(vocab.items())
    elif isinstance(vocab, list):
        vocab_pairs = vocab
    else:
        raise ValueError(f"Unknown vocab type: {type(vocab)}")

    # Keep other fields aside (everything except type/vocab/merges)
    extra = dict(state)
    extra.pop("type", None)
    extra.pop("vocab", None)
    extra.pop("merges", None)

    return model_type, vocab_pairs, merges, extra


def _filter_merges_for_vocab(merges, kept_tokens_set):
    """
    Keep only merges where both parts are present in the kept vocab.
    Supports merges as list[str] "a b" or list[tuple[str,str]].
    Returns list of tuples (str, str) for compatibility with tokenizers library.

    IMPORTANT: We need to iteratively filter merges because some merges create
    compound tokens that are then used in subsequent merges. If a compound token
    won't exist (because its constituent merge is filtered out), we must also
    filter out merges that depend on it.
    """
    if merges is None:
        return None

    # Start with all tokens in the kept vocab
    available_tokens = set(kept_tokens_set)
    filtered = []

    for m in merges:
        # Parse the merge
        if isinstance(m, str):
            parts = m.split()
            if len(parts) != 2:
                continue
            a, b = parts
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            a, b = m
        else:
            # Unknown format; skip
            continue

        # Only keep this merge if both tokens are available
        if (
            a in available_tokens
            and b in available_tokens
            and a + b in available_tokens
        ):
            # Return as tuple for tokenizers library
            filtered.append((a, b))
            # The result of this merge creates a new "available" token
            # Note: In BPE, merging "a" + "b" creates token "ab"
            available_tokens.add(a + b)

    return filtered


def trim_tokenizer_backend_model(
    tokenizer,
    kept_old_ids_in_embedding_order,
    out_dir: str,
):
    """
    Trims the tokenizer backend model so that:
      - new vocab ids are contiguous: 0..N-1
      - new ids correspond to the order in kept_old_ids_in_embedding_order
        (so tokenizer ids align with embedding rows)
      - merges are filtered if present (BPE/WordPiece)

    Returns:
      new_tokenizer (same object mutated),
      old_to_new (dict[int,int]),
      token_to_newid (dict[str,int])
    """
    # Build mapping old_id -> new_id based on embedding row order
    old_to_new = {
        int(old_id): int(new_id)
        for new_id, old_id in enumerate(kept_old_ids_in_embedding_order)
    }

    # Build token -> old_id from current tokenizer vocab
    token_to_oldid = tokenizer.get_vocab()

    # Kept tokens are those whose old id is in old_to_new
    kept_tokens = [tok for tok, oid in token_to_oldid.items() if oid in old_to_new]
    kept_tokens_set = set(kept_tokens)

    # New token -> new_id (contiguous)
    token_to_newid = {tok: old_to_new[token_to_oldid[tok]] for tok in kept_tokens}

    # --- Rebuild backend model state
    model_type, vocab_pairs, merges, extra = _get_backend_model_state(tokenizer)

    # Keep only entries in vocab that are kept, and rewrite ids to new contiguous ids
    # vocab_pairs is list[(token, old_id)] or list[(token, score)] for some models;
    # for BPE/WordPiece it’s (token, id). For Unigram it’s (token, score).
    #
    # We detect "id-like" by checking whether the second element matches token_to_oldid[token].
    # If not, we assume it's score-like (Unigram) and just keep token + score, then re-index via tokenizers model rebuild.
    #
    # In practice, Qwen BPE uses (token, id).
    new_vocab = []
    id_like = True

    # First pass: check if this is id-like or score-like
    for tok, second in vocab_pairs:
        if tok not in token_to_oldid:
            continue  # Skip tokens not in current vocab
        if second != token_to_oldid[tok]:
            # If any token doesn't match the "id" we expect, treat as score-like.
            id_like = False
            break

    # Second pass: build new_vocab with kept tokens only
    if id_like:
        for tok, _old_id in vocab_pairs:
            if tok in kept_tokens_set:
                new_vocab.append((tok, token_to_newid[tok]))
    else:
        # score-like vocab: keep (tok, score) only
        for tok, score in vocab_pairs:
            if tok in kept_tokens_set:
                new_vocab.append((tok, score))

    # Filter merges if present - only keep merges where BOTH tokens are in new vocab
    # We need to check against the actual new_vocab tokens, not just kept_tokens_set
    new_vocab_tokens = {tok for tok, _ in new_vocab}
    new_merges = _filter_merges_for_vocab(merges, new_vocab_tokens)

    # Recreate the model class with the new vocab/merges
    model_class = getattr(models, model_type)
    model_kwargs = dict(extra)
    model_kwargs["vocab"] = dict(new_vocab)
    if merges is not None:
        model_kwargs["merges"] = new_merges

    tokenizer.backend_tokenizer.model = model_class(**model_kwargs)

    try:
        tokenizer.model_max_length = getattr(
            tokenizer, "model_max_length", tokenizer.model_max_length
        )
    except Exception:
        pass

    # Save id mapping for debugging
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "old_to_new_token_id.json"), "w") as f:
        json.dump(old_to_new, f, indent=2)

    return tokenizer, old_to_new, token_to_newid


def main():
    parser = argparse.ArgumentParser(
        description="Mine token frequencies (Option B: DataLoader + bincount)"
    )
    parser.add_argument(
        "--model_path_or_name",
        type=str,
        required=True,
        help="HF path or local path to the model",
    )
    parser.add_argument(
        "--dataset_config", type=str, required=True, help="Path to YAML dataset config"
    )
    parser.add_argument(
        "--target_vocab_size",
        type=int,
        default=None,
        help="Target vocabulary size after trimming",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency threshold for inclusion",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for mining frequencies"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers for mining frequencies",
    )
    parser.add_argument(
        "--ignore_index", type=int, default=-100, help="Label ignore index to exclude"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="torch dtype name (e.g., float16, bfloat16) or auto",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map for loading the model",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Output directory to store output model and tokenizer artefacts",
    )
    args = parser.parse_args()

    model = load_model(
        args.model_path_or_name, device_map=args.device_map, dtype=args.dtype
    )
    processor = load_processor(args.model_path_or_name)

    train_dataset = build_train_dataset(args.dataset_config)

    print("Parameter Statistics of the Model before Optimization:")
    _, _, vocab_size = show_parameter(model)

    # Cache paths
    model_slug = args.model_path_or_name.replace("/", "_")
    cache_file_frequency = (
        f"{DEFAULT_CACHE_DIR}/vocab_mining/frequency.{model_slug}.json"
    )
    cache_file_vocab = f"{DEFAULT_CACHE_DIR}/vocab_mining/vocab.{model_slug}.{args.target_vocab_size}.{args.min_frequency}.json"

    os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
    os.makedirs(os.path.dirname(cache_file_vocab), exist_ok=True)

    # Mine or load frequencies of each token in the dataset
    if not os.path.exists(cache_file_frequency):
        logger.info(
            f"Frequency cache not found. Mining and caching to: {cache_file_frequency}"
        )
        counts = mine_token_frequencies(
            train_dataset=train_dataset,
            vocab_size=vocab_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ignore_index=args.ignore_index,
        )
        save_frequency_json(counts, cache_file_frequency)
    else:
        logger.info(f"Loading frequency cache from: {cache_file_frequency}")

    # Apply min_frequency and build vocab list (token_str, freq, token_id)
    freq_json = load_frequency_json(cache_file_frequency)
    kept = [
        (int(k), int(v)) for k, v in freq_json.items() if int(v) >= args.min_frequency
    ]

    # Sort by descending frequency
    kept.sort(key=lambda x: x[1], reverse=True)

    if args.target_vocab_size is not None and args.target_vocab_size > 0:
        print(f"Truncating vocabulary to size: {args.target_vocab_size}")
        kept = kept[: args.target_vocab_size]

    vocab_list = [
        (processor.tokenizer.convert_ids_to_tokens(token_id), freq, token_id)
        for token_id, freq in kept
    ]

    # Save vocab list
    with open(cache_file_vocab, "w") as f:
        new_vocab = {x[0]: x[2] for x in vocab_list}
        json.dump(new_vocab, f, ensure_ascii=False)

    logger.info(f"Saved vocab list: {cache_file_vocab}")
    logger.info(
        f"Kept tokens (>= {args.min_frequency}, capped to {args.target_vocab_size}): {pretty(len(vocab_list))}"
    )

    # Quick summary
    if vocab_list:
        logger.info(
            f"Most frequent token: {vocab_list[0][0]} (id={vocab_list[0][2]}) freq={pretty(vocab_list[0][1])}"
        )
        logger.info(
            f"Least frequent kept token: {vocab_list[-1][0]} (id={vocab_list[-1][2]}) freq={pretty(vocab_list[-1][1])}"
        )

    vocab = dict(
        zip(processor.tokenizer.all_special_tokens, processor.tokenizer.all_special_ids)
    )

    # keep all the merges or tokens with the newline Ċ token
    vocab = {
        k: v for k, v in processor.tokenizer.get_vocab().items() if "Ċ" in k
    } | vocab

    vocab.update(new_vocab)

    new_vocab_id = sorted(vocab.values())
    new_vocab = list(vocab.keys())
    print(
        f"trimming vocabulary: {pretty(len(processor.tokenizer.vocab))} (original) -> {pretty(len(new_vocab_id))} (target)"
    )

    # get the current embeddings
    inp_embeddings = model.language_model.get_input_embeddings()

    model.language_model.set_input_embeddings(
        torch.nn.Embedding.from_pretrained(inp_embeddings.weight[new_vocab_id])
    )

    model.language_model.config.vocab_size = len(new_vocab_id)
    model.config.vocab_size = len(new_vocab_id)

    model.resize_token_embeddings(model.config.vocab_size)

    print("Parameter Statistics of the Model before Optimization:")
    _, _, vocab_size = show_parameter(model)

    # ---------------------------------------------------------------------
    # Trim tokenizer backend model to match pruned embeddings (contiguous ids)
    # ---------------------------------------------------------------------
    os.makedirs(args.output_directory, exist_ok=True)

    print("updating tokenizer ...")

    # IMPORTANT: kept ids in the exact order of embedding rows
    # list[int], sorted by old id
    kept_old_ids_in_embedding_order = new_vocab_id

    processor.tokenizer, old_to_new, token_to_newid = trim_tokenizer_backend_model(
        processor.tokenizer,
        kept_old_ids_in_embedding_order=kept_old_ids_in_embedding_order,
        out_dir=args.output_directory,
    )

    # Save
    processor.save_pretrained(args.output_directory)
    model.save_pretrained(args.output_directory)

    print(f"Saved pruned model + processor to: {args.output_directory}")
    print(
        f"Saved old->new id map to: {os.path.join(args.output_directory, 'old_to_new_token_id.json')}"
    )


if __name__ == "__main__":
    main()
