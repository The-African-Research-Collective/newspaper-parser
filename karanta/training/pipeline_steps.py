import logging
import base64
import json
import yaml
import torch
import numpy as np

from dataclasses import dataclass
from abc import ABC
from jinja2 import Template

from PIL import Image
from io import BytesIO
from typing import Any

from karanta.training.utils import SingleDatapoint
from karanta.prompts.anchor import get_anchor_text
from karanta.data.process_pdf_utils import render_pdf_to_base64png

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BasePipelineStep(ABC):
    """This is the base class for all the pipeline steps."""

    def __call__(self, *args, **kwargs):
        """Call the step with the given arguments."""
        pass


@dataclass(frozen=True, slots=True)
class PDF2ImageStep(BasePipelineStep):
    """Pipeline step that renders PDF to image."""

    target_longest_image_dim: int

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """Render PDF to image."""
        # # Render PDF to image
        base64_png = render_pdf_to_base64png(
            str(sample.pdf_path),
            page_num=1,
            target_longest_image_dim=self.target_longest_image_dim,
        )
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))

        # # Update sample
        sample.image = image

        return sample


@dataclass(frozen=True, slots=True)
class JSONOutputFormat(BasePipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        page_data_list = sample.page_data

        try:
            if isinstance(page_data_list[0], str):
                page_data_list = [json.loads(page_data_list[0])]
                
            sample.response = json.dumps(
                [
                    {
                        "primary_language": page_data["primary_language"],
                        "is_rotation_valid": page_data["is_rotation_valid"],
                        "rotation_correction": page_data["rotation_correction"],
                        "is_table": page_data["is_table"],
                        "is_diagram": page_data["is_diagram"],
                        "natural_text": page_data["natural_text"],
                    }
                    for page_data in page_data_list
                ],
                ensure_ascii=False,
            )
            return sample
        except:
            sample.response = None
            return sample


@dataclass(frozen=True, slots=True)
class FetchPageData(BasePipelineStep):
    """Fetch the page data from the JSON file and store it in the sample."""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        with open(sample.json_path, "r", encoding="utf-8") as f:
            output_result = json.loads(json.loads(f.read())["result"]["text"])

        sample.page_data = output_result
        return sample


@dataclass(frozen=True, slots=True)
class FetchMultipageData(BasePipelineStep):
    """Similar to FetchPageData but this is for multipage format"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        with open(sample.json_path, "r", encoding="utf-8") as f:
            output_result = json.loads(f.read())["generation"]["pages"]

        sample.page_data = output_result
        return sample


@dataclass(frozen=True, slots=True)
class StaticLengthDocumentAnchoring(BasePipelineStep):
    target_anchor_text_len: int

    """Pipeline step that runs document anchoring on the PDF and puts in the data to be used by later prompting stages"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        anchor_text = get_anchor_text(
            str(sample.pdf_path),
            page=1,
            pdf_engine="pdfreport",
            target_length=self.target_anchor_text_len,
        )
        sample.anchor_text = anchor_text
        return sample


@dataclass(frozen=True, slots=True)
class FinetuningPrompt(BasePipelineStep):
    """Applies the standard fine tuning prompt"""

    prompt_path: str

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        image_page = True

        if sample.anchor_text and len(sample.anchor_text.split("\n")) > 10:
            image_page = False
        else:
            image_page = True

        with open(self.prompt_path, "r") as stream:
            prompt_template_dict = yaml.safe_load(stream)

            if image_page:
                prompt_template_dict["system"] = prompt_template_dict[
                    "olmo_ocr_system_prompt_no_anchor"
                ]
                sample.instruction_prompt = prompt_template_dict["system"]
            else:
                prompt_template_dict["system"] = Template(
                    prompt_template_dict["olmo_ocr_system_prompt"]
                )
                sample.instruction_prompt = prompt_template_dict["system"].render(
                    {"base_text": sample.anchor_text}
                )
        return sample


@dataclass(frozen=True, slots=True)
class InstructUserMessages(BasePipelineStep):
    """Creates instruction-following messages format for training."""

    prompt_first: bool = False

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        # Prepare messages
        if self.prompt_first:
            messages = {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.instruction_prompt},
                    {"type": "image", "image": sample.image},
                ],
            }
        else:
            messages = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": sample.instruction_prompt},
                ],
            }

        sample.user_messages = messages

        return sample


@dataclass(frozen=True, slots=True)
class Tokenizer(BasePipelineStep):
    """
    Tokenizes document images and creates training labels for OCR (Optical Character Recognition) models.

    This class is designed for training vision-language models that can extract and structure text
    from document images. It processes document images along with user instructions and generates
    structured output in formats like XML, JSON, or YAML.

    The key insight for OCR training is that we want the model to:
    1. See the full context (document image + extraction instruction + structured output)
    2. Only be penalized for mistakes in generating the structured text output
    3. Learn to convert visual document content into machine-readable structured formats

    Attributes:
        processor: The multimodal processor (e.g., AutoProcessor) that handles document images and text
        masking_index: Value used to mask tokens that shouldn't contribute to loss (typically -100)
        end_of_message_token: Token that marks the end of text extraction (defaults to Qwen format)
    """

    processor: Any  # The model processor (e.g., AutoProcessor)
    max_length: int
    masking_index: int = (
        -100
    )  # Standard PyTorch value for ignoring tokens in loss calculation
    end_of_message_token: str = "<|im_end|>"  # Configurable, defaults to Qwen format

    def _pad_sequence(self, sequence, pad_value: int) -> np.ndarray:
        """
        Pad a sequence to the specified maximum length.

        Args:
            sequence: The sequence to pad
            max_length: Target length after padding
            pad_value: Value to use for padding

        Returns:
            Padded sequence
        """
        if len(sequence) < self.max_length:
            extra_length = self.max_length - len(sequence)
            padding = torch.full((extra_length,), pad_value, dtype=sequence.dtype)

            if self.processor.tokenizer.padding_side == "right":
                return torch.cat([sequence, padding])
            else:
                return torch.cat([padding, sequence])

        elif len(sequence) > self.max_length:
            return sequence[: self.max_length]

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """
        Tokenize document images and create labels for OCR training.

        This method transforms a training sample into the format needed by the OCR model:
        - Processes document images with extraction instructions
        - Creates proper attention masks
        - Sets up labels with masking to only train on structured output generation

        Args:
            sample: A SingleDatapoint containing user_messages (document image + instruction)
                   and response (structured text output in XML/JSON/YAML)

        Returns:
            The same sample with model_inputs populated with tokenized data
        """

        # === DATA EXTRACTION ===
        # Extract the core components of our OCR training sample
        user_messages = (
            sample.user_messages
        )  # User's instruction + document image to process
        response = (
            sample.response
        )  # Target structured output (XML/JSON/YAML) the model should learn to generate

        # === CHAT TEMPLATE APPLICATION ===
        # Convert user instruction into the model's expected format for document processing
        # add_generation_prompt=True adds special tokens that signal "start extracting text here"
        # tokenize=False keeps it as text for now (we'll tokenize everything together later)
        text = self.processor.apply_chat_template(
            [user_messages],  # Wrap in list as expected by the template
            tokenize=False,  # Keep as text, don't tokenize yet
            add_generation_prompt=True,  # Add generation tokens depending on the model
        )

        # === DOCUMENT IMAGE EXTRACTION ===
        # Find the document image in the user's message content
        # For OCR training, this will be a document image (PDF page)
        # This assumes exactly one document image per training sample
        main_image = None
        for usg_msg in user_messages["content"]:
            if "image" in usg_msg:
                main_image = usg_msg["image"]  # The document image to extract text from
                break

        # Ensure we found a document image (this OCR tokenizer expects document input)
        assert main_image is not None, (
            "Expected to find a document image in user messages"
        )

        # === INPUT TOKENIZATION ===
        # Process both the extraction instruction and document image using the multimodal processor
        # This creates:
        # - input_ids: tokenized instruction text (e.g., "Extract text as JSON")
        # - pixel_values: processed document image features (text regions, layout, etc.)
        # - attention_mask: which tokens to pay attention to
        inputs = self.processor(
            text=[text],  # The formatted extraction instruction
            images=[main_image],  # The document image to process
            padding=False,  # Pad sequences to same length
            return_tensors="pt",  # Return NumPy arrays
        )

        # === STRUCTURED OUTPUT TOKENIZATION ===
        # Tokenize the target structured text output (XML/JSON/YAML format)
        # This is what the model should learn to generate from the document image
        # Examples: JSON with extracted fields, XML with document structure, YAML with metadata
        labels = self.processor(text=[response], padding=False, return_tensors="pt")

        # === END-OF-EXTRACTION TOKEN HANDLING ===
        # Add special token to mark where the model should stop generating structured output
        # This helps the model know when it has completed the text extraction task
        # add_special_tokens=False prevents adding extra BOS/EOS tokens
        end_tokens = self.processor.tokenizer(
            self.end_of_message_token, add_special_tokens=False
        )["input_ids"]

        # Convert to same data type as other tokens for concatenation
        end_tokens = torch.tensor(end_tokens, dtype=inputs.input_ids.dtype)

        # === LABEL CONCATENATION WITH END TOKEN ===
        # Handle edge case where structured output might be empty (rare for OCR tasks)
        if labels["input_ids"].shape[1] == 0:
            labels_input_ids_0 = torch.tensor([], dtype=inputs.input_ids.dtype)
        else:
            labels_input_ids_0 = labels["input_ids"][0].type(inputs.input_ids.dtype)

        # Append end token to the structured output labels
        labels["input_ids"] = torch.cat([labels_input_ids_0, end_tokens])
        labels["input_ids"] = torch.unsqueeze(
            labels["input_ids"], axis=0
        )  # Add batch dimension back

        # === SEQUENCE CONSTRUCTION ===
        # Create the full training sequence: [extraction_instruction] + [structured_output] + [end_token]
        # The model will see this entire sequence but only be trained to generate the structured output part
        # Example: "Extract as JSON" + document_image -> {"name": "John", "address": "123 Main St"} + <|im_end|>
        input_ids = torch.cat([inputs.input_ids[0], labels.input_ids[0]], axis=0)

        # === ATTENTION MASK CREATION ===
        # Create attention mask - all tokens participate in attention
        # The model needs to attend to both the instruction and document image features
        # to understand what type of extraction is requested and what content to extract
        attention_mask = torch.ones_like(input_ids)

        # === TRAINING LABEL CREATION (CRITICAL FOR OCR LEARNING) ===
        # Create labels for training with proper masking:
        # - Fill entire sequence with masking_index (-100) initially
        # - Only the structured output portion gets actual token IDs as labels
        # This ensures the model only learns to generate structured text output, not repeat the instruction
        # The model learns: document_image + instruction -> structured_output
        labels_full = torch.full_like(input_ids, fill_value=self.masking_index)

        # Unmask only the structured output portion (everything after the instruction prompt)
        # The model will be penalized only for mistakes in generating the XML/JSON/YAML output
        labels_full[len(inputs.input_ids[0]) :] = labels.input_ids[0]

        # === OUTPUT ASSEMBLY ====
        # Package all processed data into the format expected by the OCR model trainer
        sample.model_inputs["input_ids"] = (
            input_ids  # The full tokenized sequence (instruction + output)
        )
        sample.model_inputs["attention_mask"] = (
            attention_mask  # Which tokens to attend to (all of them)
        )
        sample.model_inputs["labels"] = (
            labels_full  # Training targets (masked for instruction portion)
        )
        sample.model_inputs["pixel_values"] = (
            inputs.pixel_values
        )  # Processed document image features

        # === OPTIONAL DOCUMENT IMAGE METADATA ===
        # Some models need additional image processing metadata for document understanding
        # image_grid_thw typically contains height, width, and other document image dimensions
        # This helps with layout understanding and text region detection

        if hasattr(inputs, "image_grid_thw"):
            sample.model_inputs["image_grid_thw"] = inputs.image_grid_thw[0]

        return sample
import logging
import base64
import json
import yaml
import torch
import numpy as np

from dataclasses import dataclass
from abc import ABC
from jinja2 import Template

from PIL import Image
from io import BytesIO
from typing import Any

from karanta.training.utils import SingleDatapoint
from karanta.prompts.anchor import get_anchor_text
from karanta.data.process_pdf_utils import render_pdf_to_base64png

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BasePipelineStep(ABC):
    """This is the base class for all the pipeline steps."""

    def __call__(self, *args, **kwargs):
        """Call the step with the given arguments."""
        pass


@dataclass(frozen=True, slots=True)
class PDF2ImageStep(BasePipelineStep):
    """Pipeline step that renders PDF to image."""

    target_longest_image_dim: int

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """Render PDF to image."""
        # # Render PDF to image
        base64_png = render_pdf_to_base64png(
            str(sample.pdf_path),
            page_num=1,
            target_longest_image_dim=self.target_longest_image_dim,
        )
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))

        # # Update sample
        sample.image = image

        return sample


@dataclass(frozen=True, slots=True)
class JSONOutputFormat(BasePipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        page_data_list = sample.page_data

        try:
            if isinstance(page_data_list[0], str):
                page_data_list = [json.loads(page_data_list[0])]
                
            sample.response = json.dumps(
                [
                    {
                        "primary_language": page_data["primary_language"],
                        "is_rotation_valid": page_data["is_rotation_valid"],
                        "rotation_correction": page_data["rotation_correction"],
                        "is_table": page_data["is_table"],
                        "is_diagram": page_data["is_diagram"],
                        "natural_text": page_data["natural_text"],
                    }
                    for page_data in page_data_list
                ],
                ensure_ascii=False,
            )
            return sample
        except:
            sample.response = None
            return sample


@dataclass(frozen=True, slots=True)
class FetchPageData(BasePipelineStep):
    """Fetch the page data from the JSON file and store it in the sample."""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        with open(sample.json_path, "r", encoding="utf-8") as f:
            output_result = json.loads(json.loads(f.read())["result"]["text"])

        sample.page_data = output_result
        return sample


@dataclass(frozen=True, slots=True)
class FetchMultipageData(BasePipelineStep):
    """Similar to FetchPageData but this is for multipage format"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        with open(sample.json_path, "r", encoding="utf-8") as f:
            output_result = json.loads(f.read())["generation"]["pages"]

        sample.page_data = output_result
        return sample


@dataclass(frozen=True, slots=True)
class StaticLengthDocumentAnchoring(BasePipelineStep):
    target_anchor_text_len: int

    """Pipeline step that runs document anchoring on the PDF and puts in the data to be used by later prompting stages"""

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        anchor_text = get_anchor_text(
            str(sample.pdf_path),
            page=1,
            pdf_engine="pdfreport",
            target_length=self.target_anchor_text_len,
        )
        sample.anchor_text = anchor_text
        return sample


@dataclass(frozen=True, slots=True)
class FinetuningPrompt(BasePipelineStep):
    """Applies the standard fine tuning prompt"""

    prompt_path: str

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        image_page = True

        if sample.anchor_text and len(sample.anchor_text.split("\n")) > 10:
            image_page = False
        else:
            image_page = True

        with open(self.prompt_path, "r") as stream:
            prompt_template_dict = yaml.safe_load(stream)

            if image_page:
                prompt_template_dict["system"] = prompt_template_dict[
                    "olmo_ocr_system_prompt_no_anchor"
                ]
                sample.instruction_prompt = prompt_template_dict["system"]
            else:
                prompt_template_dict["system"] = Template(
                    prompt_template_dict["olmo_ocr_system_prompt"]
                )
                sample.instruction_prompt = prompt_template_dict["system"].render(
                    {"base_text": sample.anchor_text}
                )
        return sample


@dataclass(frozen=True, slots=True)
class InstructUserMessages(BasePipelineStep):
    """Creates instruction-following messages format for training."""

    prompt_first: bool = False

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        # Prepare messages
        if self.prompt_first:
            messages = {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample.instruction_prompt},
                    {"type": "image", "image": sample.image},
                ],
            }
        else:
            messages = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": sample.instruction_prompt},
                ],
            }

        sample.user_messages = messages

        return sample


@dataclass(frozen=True, slots=True)
class Tokenizer(BasePipelineStep):
    """
    Tokenizes document images and creates training labels for OCR (Optical Character Recognition) models.

    This class is designed for training vision-language models that can extract and structure text
    from document images. It processes document images along with user instructions and generates
    structured output in formats like XML, JSON, or YAML.

    The key insight for OCR training is that we want the model to:
    1. See the full context (document image + extraction instruction + structured output)
    2. Only be penalized for mistakes in generating the structured text output
    3. Learn to convert visual document content into machine-readable structured formats

    Attributes:
        processor: The multimodal processor (e.g., AutoProcessor) that handles document images and text
        masking_index: Value used to mask tokens that shouldn't contribute to loss (typically -100)
        end_of_message_token: Token that marks the end of text extraction (defaults to Qwen format)
    """

    processor: Any  # The model processor (e.g., AutoProcessor)
    max_length: int
    masking_index: int = (
        -100
    )  # Standard PyTorch value for ignoring tokens in loss calculation
    end_of_message_token: str = "<|im_end|>"  # Configurable, defaults to Qwen format

    def _pad_sequence(self, sequence, pad_value: int) -> np.ndarray:
        """
        Pad a sequence to the specified maximum length.

        Args:
            sequence: The sequence to pad
            max_length: Target length after padding
            pad_value: Value to use for padding

        Returns:
            Padded sequence
        """
        if len(sequence) < self.max_length:
            extra_length = self.max_length - len(sequence)
            padding = torch.full((extra_length,), pad_value, dtype=sequence.dtype)

            if self.processor.tokenizer.padding_side == "right":
                return torch.cat([sequence, padding])
            else:
                return torch.cat([padding, sequence])

        elif len(sequence) > self.max_length:
            return sequence[: self.max_length]

    def __call__(self, sample: SingleDatapoint) -> SingleDatapoint:
        """
        Tokenize document images and create labels for OCR training.

        This method transforms a training sample into the format needed by the OCR model:
        - Processes document images with extraction instructions
        - Creates proper attention masks
        - Sets up labels with masking to only train on structured output generation

        Args:
            sample: A SingleDatapoint containing user_messages (document image + instruction)
                   and response (structured text output in XML/JSON/YAML)

        Returns:
            The same sample with model_inputs populated with tokenized data
        """

        # === DATA EXTRACTION ===
        # Extract the core components of our OCR training sample
        user_messages = (
            sample.user_messages
        )  # User's instruction + document image to process
        response = (
            sample.response
        )  # Target structured output (XML/JSON/YAML) the model should learn to generate

        # === CHAT TEMPLATE APPLICATION ===
        # Convert user instruction into the model's expected format for document processing
        # add_generation_prompt=True adds special tokens that signal "start extracting text here"
        # tokenize=False keeps it as text for now (we'll tokenize everything together later)
        text = self.processor.apply_chat_template(
            [user_messages],  # Wrap in list as expected by the template
            tokenize=False,  # Keep as text, don't tokenize yet
            add_generation_prompt=True,  # Add generation tokens depending on the model
        )

        # === DOCUMENT IMAGE EXTRACTION ===
        # Find the document image in the user's message content
        # For OCR training, this will be a document image (PDF page)
        # This assumes exactly one document image per training sample
        main_image = None
        for usg_msg in user_messages["content"]:
            if "image" in usg_msg:
                main_image = usg_msg["image"]  # The document image to extract text from
                break

        # Ensure we found a document image (this OCR tokenizer expects document input)
        assert main_image is not None, (
            "Expected to find a document image in user messages"
        )

        # === INPUT TOKENIZATION ===
        # Process both the extraction instruction and document image using the multimodal processor
        # This creates:
        # - input_ids: tokenized instruction text (e.g., "Extract text as JSON")
        # - pixel_values: processed document image features (text regions, layout, etc.)
        # - attention_mask: which tokens to pay attention to
        inputs = self.processor(
            text=[text],  # The formatted extraction instruction
            images=[main_image],  # The document image to process
            padding=False,  # Pad sequences to same length
            return_tensors="pt",  # Return NumPy arrays
        )

        # === STRUCTURED OUTPUT TOKENIZATION ===
        # Tokenize the target structured text output (XML/JSON/YAML format)
        # This is what the model should learn to generate from the document image
        # Examples: JSON with extracted fields, XML with document structure, YAML with metadata
        labels = self.processor(text=[response], padding=False, return_tensors="pt")

        # === END-OF-EXTRACTION TOKEN HANDLING ===
        # Add special token to mark where the model should stop generating structured output
        # This helps the model know when it has completed the text extraction task
        # add_special_tokens=False prevents adding extra BOS/EOS tokens
        end_tokens = self.processor.tokenizer(
            self.end_of_message_token, add_special_tokens=False
        )["input_ids"]

        # Convert to same data type as other tokens for concatenation
        end_tokens = torch.tensor(end_tokens, dtype=inputs.input_ids.dtype)

        # === LABEL CONCATENATION WITH END TOKEN ===
        # Handle edge case where structured output might be empty (rare for OCR tasks)
        if labels["input_ids"].shape[1] == 0:
            labels_input_ids_0 = torch.tensor([], dtype=inputs.input_ids.dtype)
        else:
            labels_input_ids_0 = labels["input_ids"][0].type(inputs.input_ids.dtype)

        # Append end token to the structured output labels
        labels["input_ids"] = torch.cat([labels_input_ids_0, end_tokens])
        labels["input_ids"] = torch.unsqueeze(
            labels["input_ids"], axis=0
        )  # Add batch dimension back

        # === SEQUENCE CONSTRUCTION ===
        # Create the full training sequence: [extraction_instruction] + [structured_output] + [end_token]
        # The model will see this entire sequence but only be trained to generate the structured output part
        # Example: "Extract as JSON" + document_image -> {"name": "John", "address": "123 Main St"} + <|im_end|>
        input_ids = torch.cat([inputs.input_ids[0], labels.input_ids[0]], axis=0)

        # === ATTENTION MASK CREATION ===
        # Create attention mask - all tokens participate in attention
        # The model needs to attend to both the instruction and document image features
        # to understand what type of extraction is requested and what content to extract
        attention_mask = torch.ones_like(input_ids)

        # === TRAINING LABEL CREATION (CRITICAL FOR OCR LEARNING) ===
        # Create labels for training with proper masking:
        # - Fill entire sequence with masking_index (-100) initially
        # - Only the structured output portion gets actual token IDs as labels
        # This ensures the model only learns to generate structured text output, not repeat the instruction
        # The model learns: document_image + instruction -> structured_output
        labels_full = torch.full_like(input_ids, fill_value=self.masking_index)

        # Unmask only the structured output portion (everything after the instruction prompt)
        # The model will be penalized only for mistakes in generating the XML/JSON/YAML output
        labels_full[len(inputs.input_ids[0]) :] = labels.input_ids[0]

        # === OUTPUT ASSEMBLY ====
        # Package all processed data into the format expected by the OCR model trainer
        sample.model_inputs["input_ids"] = (
            input_ids  # The full tokenized sequence (instruction + output)
        )
        sample.model_inputs["attention_mask"] = (
            attention_mask  # Which tokens to attend to (all of them)
        )
        sample.model_inputs["labels"] = (
            labels_full  # Training targets (masked for instruction portion)
        )
        sample.model_inputs["pixel_values"] = (
            inputs.pixel_values
        )  # Processed document image features

        # === OPTIONAL DOCUMENT IMAGE METADATA ===
        # Some models need additional image processing metadata for document understanding
        # image_grid_thw typically contains height, width, and other document image dimensions
        # This helps with layout understanding and text region detection

        if hasattr(inputs, "image_grid_thw"):
            sample.model_inputs["image_grid_thw"] = inputs.image_grid_thw[0]

        return sample

