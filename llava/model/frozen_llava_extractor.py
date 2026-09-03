"""
Frozen LLaVA Feature Extractor

Loads a pretrained LLaVA model (with optional LoRA) and extracts hidden
states before the language model head.

Supports:
  - Hugging Face LLaVA, e.g. llava-hf/llava-1.5-7b-hf
  - Original LLaVA checkpoints

For Hugging Face LLaVA:
  vision_tower          -> model.vision_tower
  projector             -> model.multi_modal_projector
  image preprocessing   -> LlavaProcessor
  image input           -> pixel_values

For original LLaVA:
  vision tower          -> model.get_vision_tower()
  projector             -> model.get_model().mm_projector
  image preprocessing   -> original LLaVA image processor
  image input           -> images
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers import AutoProcessor

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from peft import PeftModel


class FrozenLLaVAExtractor(nn.Module):
    """
    Frozen LLaVA model for extracting hidden states as conditioning signals.

    Args:
        llava_base_path:
            Path to the base LLaVA model.

        llava_lora_path:
            Optional LoRA adapter path.

        device:
            Target device, e.g. "cuda:0".

        load_8bit:
            Load language model using bitsandbytes 8-bit quantization.

        load_4bit:
            Load language model using bitsandbytes 4-bit quantization.

        merge_lora:
            Merge LoRA into the base model after loading.
    """

    def __init__(
        self,
        llava_base_path: str,
        llava_lora_path: Optional[str] = None,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        merge_lora: bool = True,
    ):
        super().__init__()

        self.device = device
        self.llava_base_path = llava_base_path
        self.llava_lora_path = llava_lora_path

        print(f"Loading frozen LLaVA from: {llava_base_path}")

        if llava_lora_path:
            print(f"  + LoRA weights from: {llava_lora_path}")

        # ---------------------------------------------------------
        # Load model using your existing LLaVA builder
        # ---------------------------------------------------------

        model_name = get_model_name_from_path(llava_base_path)

        target_device_map = (
            {"": device}
            if isinstance(device, str) and device.startswith("cuda")
            else None
        )

        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            llava_base_path,
            None,
            model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device_map=(
                target_device_map
                if target_device_map is not None
                else device
            ),
            device=device,
        )

        # ---------------------------------------------------------
        # Detect HF LLaVA
        #
        # HF LLaVA has:
        #   model.vision_tower
        #   model.multi_modal_projector
        #
        # Original LLaVA does not.
        # ---------------------------------------------------------

        self.is_hf_llava = hasattr(
            self.model,
            "multi_modal_projector",
        )

        if self.is_hf_llava:
            print("Detected Hugging Face LLaVA architecture")

            # The HF processor owns:
            #   tokenizer
            #   image_processor
            #
            # This is intentionally separate from the original
            # LLaVA image_processor returned by builder.py.
            self.processor = AutoProcessor.from_pretrained(
                llava_base_path
            )

            # Prefer processor versions of these.
            self.tokenizer = self.processor.tokenizer
            self.image_processor = self.processor.image_processor

            print(
                f"  vision_tower: {type(self.model.vision_tower)}"
            )
            print(
                f"  projector: {type(self.model.multi_modal_projector)}"
            )
            print(
                f"  processor: {type(self.processor)}"
            )

        else:
            print("Detected original LLaVA architecture")

            self.processor = None

        # ---------------------------------------------------------
        # Load LoRA
        # ---------------------------------------------------------

        if llava_lora_path:
            print("Loading LoRA adapter...")

            self.model = PeftModel.from_pretrained(
                self.model,
                llava_lora_path,
            )

            # -----------------------------------------------------
            # Load non-LoRA trainables.
            #
            # NOTE:
            # If this is a LoRA checkpoint produced by original
            # LLaVA, the parameter names may not match HF LLaVA.
            # strict=False is therefore intentional.
            # -----------------------------------------------------

            non_lora_path = os.path.join(
                llava_lora_path,
                "non_lora_trainables.bin",
            )

            if os.path.exists(non_lora_path):
                print("Loading non-LoRA trainables...")

                non_lora_weights = torch.load(
                    non_lora_path,
                    map_location="cpu",
                )

                # Remove common wrapper prefixes if necessary.
                cleaned_weights = {}

                for key, value in non_lora_weights.items():
                    new_key = key

                    # Common original-LLaVA checkpoint prefixes.
                    if new_key.startswith("base_model."):
                        new_key = new_key[len("base_model."):]

                    if new_key.startswith("model."):
                        # Don't blindly remove "model." for HF models.
                        # Keep the original key if it already matches.
                        if new_key not in self.model.state_dict():
                            candidate = new_key[len("model."):]
                            if candidate in self.model.state_dict():
                                new_key = candidate

                    cleaned_weights[new_key] = value

                if not load_8bit and not load_4bit:
                    cleaned_weights = {
                        k: v.to(torch.float16)
                        if torch.is_floating_point(v)
                        else v
                        for k, v in cleaned_weights.items()
                    }

                missing, unexpected = self.model.load_state_dict(
                    cleaned_weights,
                    strict=False,
                )

                if missing:
                    print(
                        f"  non-LoRA missing keys: {len(missing)}"
                    )

                if unexpected:
                    print(
                        f"  non-LoRA unexpected keys: {len(unexpected)}"
                    )

            else:
                print(
                    "Warning: non_lora_trainables.bin not found. "
                    "Projector may not contain the fine-tuned weights."
                )

            # -----------------------------------------------------
            # Merge LoRA
            # -----------------------------------------------------

            if merge_lora:
                try:
                    print(
                        "Merging LoRA adapters into base model..."
                    )

                    self.model = self.model.merge_and_unload()

                    print("✓ LoRA merged")

                except Exception as e:
                    print(
                        "Warning: merge_and_unload failed "
                        f"({e}); continuing with PEFT wrappers."
                    )

        # ---------------------------------------------------------
        # Set device
        # ---------------------------------------------------------

        try:
            if (
                isinstance(self.device, str)
                and self.device.startswith("cuda")
                and torch.cuda.is_available()
            ):
                try:
                    local_idx = (
                        int(self.device.split(":")[1])
                        if ":" in self.device
                        else 0
                    )

                    torch.cuda.set_device(local_idx)

                except Exception:
                    pass

            self.model.to(self.device)

        except Exception:
            # Needed if the model has been sharded.
            pass

        # ---------------------------------------------------------
        # Freeze
        # ---------------------------------------------------------

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("✓ LLaVA model loaded and frozen")

    # =============================================================
    # IMAGE CONVERSION
    # =============================================================

    def _tensor_to_pil(self, image):
        """
        Convert a CHW image tensor to PIL.

        This is intended for ordinary [0,1] image tensors.

        If a tensor has already been normalized using CLIP mean/std,
        pass the original PIL image instead whenever possible.
        """

        from PIL import Image

        image = image.detach().float().cpu()

        if image.ndim != 3:
            raise ValueError(
                f"Expected CHW image, got {image.shape}"
            )

        if image.shape[0] != 3:
            raise ValueError(
                f"Expected 3-channel image, got {image.shape}"
            )

        image = image.clamp(0.0, 1.0)

        image = (
            image
            .permute(1, 2, 0)
            .mul(255.0)
            .byte()
            .numpy()
        )

        return Image.fromarray(image)

    # =============================================================
    # PREPARE INPUTS
    # =============================================================

    def prepare_inputs(
        self,
        images,
        prompts: list,
        conv_mode: str = "llava_v1",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs.

        HF LLaVA:
            Uses LlavaProcessor for both text and images.

        Original LLaVA:
            Uses the original tokenizer_image_token and image
            processor.

        Returns:
            input_ids:
                Tensor of shape (B, L)

            image_tensors:
                For HF LLaVA this is pixel_values.
                For original LLaVA this is images.
        """

        from PIL import Image

        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"number of prompts ({len(prompts)})"
            )

        # =========================================================
        # HF LLAVA
        # =========================================================

        if self.is_hf_llava:

            pil_images = []

            if isinstance(images, list):

                for image in images:

                    if isinstance(image, Image.Image):
                        pil_images.append(image)

                    elif torch.is_tensor(image):
                        pil_images.append(
                            self._tensor_to_pil(image)
                        )

                    else:
                        raise TypeError(
                            f"Unsupported image type: "
                            f"{type(image)}"
                        )

            elif torch.is_tensor(images):

                if images.ndim != 4:
                    raise ValueError(
                        "Expected image tensor with shape "
                        f"(B, C, H, W), got {images.shape}"
                    )

                for image in images:
                    pil_images.append(
                        self._tensor_to_pil(image)
                    )

            else:
                raise TypeError(
                    f"Unsupported images type: {type(images)}"
                )

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # HF LLaVA expects the image placeholder in the text.
            #
            # The processor then creates:
            #   input_ids
            #   attention_mask
            #   pixel_values
            #
            # We only return the first and third here because your
            # existing extractor interface expects two tensors.
            # -----------------------------------------------------

            hf_prompts = []

            for prompt in prompts:

                # Avoid adding <image> twice.
                if "<image>" in prompt:
                    hf_prompt = prompt
                else:
                    hf_prompt = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + prompt
                    )

                hf_prompts.append(hf_prompt)

            inputs = self.processor(
                text=hf_prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )

            input_ids = inputs["input_ids"].to(
                self.device
            )

            pixel_values = inputs["pixel_values"].to(
                self.device
            )

            # Keep attention mask for the forward pass.
            self._attention_mask = inputs.get(
                "attention_mask",
                None,
            )

            if self._attention_mask is not None:
                self._attention_mask = (
                    self._attention_mask.to(self.device)
                )

            return input_ids, pixel_values

        # =========================================================
        # ORIGINAL LLAVA
        # =========================================================

        if isinstance(images, list):

            processed_images = []

            for image in images:

                if isinstance(image, Image.Image):

                    image_tensor = (
                        self.image_processor.preprocess(
                            image,
                            return_tensors="pt",
                        )["pixel_values"][0]
                    )

                    processed_images.append(
                        image_tensor
                    )

                elif torch.is_tensor(image):

                    processed_images.append(image)

                else:

                    raise TypeError(
                        f"Unsupported image type: "
                        f"{type(image)}"
                    )

            images = torch.stack(
                processed_images
            ).to(self.device)

        elif torch.is_tensor(images):

            if images.device != self.device:
                images = images.to(self.device)

        else:

            raise TypeError(
                f"Unsupported images type: {type(images)}"
            )

        input_ids_list = []

        for prompt in prompts:

            if getattr(
                self.model.config,
                "mm_use_im_start_end",
                False,
            ):

                prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                )

            else:

                prompt = (
                    DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + prompt
                )

            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )

            input_ids_list.append(input_ids)

        max_len = max(
            ids.size(0)
            for ids in input_ids_list
        )

        input_ids_padded = []

        for ids in input_ids_list:

            padding = torch.full(
                (
                    max_len
                    - ids.size(0),
                ),
                self.tokenizer.pad_token_id,
                dtype=ids.dtype,
            )

            input_ids_padded.append(
                torch.cat(
                    [
                        ids,
                        padding,
                    ]
                )
            )

        input_ids = torch.stack(
            input_ids_padded
        ).to(self.device)

        self._attention_mask = (
            input_ids
            != self.tokenizer.pad_token_id
        ).long()

        return input_ids, images

    # =============================================================
    # HIDDEN STATES
    # =============================================================

    @torch.no_grad()
    def extract_hidden_states(
        self,
        images,
        prompts: list,
    ) -> torch.Tensor:
        """
        Extract hidden states before lm_head.

        Returns:
            Tensor of shape approximately:

                (B, sequence_length, 4096)
        """

        input_ids, image_tensors = (
            self.prepare_inputs(
                images,
                prompts,
            )
        )

        # ---------------------------------------------------------
        # HF LLAVA
        # ---------------------------------------------------------

        if self.is_hf_llava:

            pixel_values = image_tensors

            # HF CLIP expects floating point pixel values.
            if pixel_values.dtype != torch.float16:
                pixel_values = pixel_values.to(
                    dtype=torch.float16
                )

            pixel_values = pixel_values.contiguous()

            attention_mask = getattr(
                self,
                "_attention_mask",
                None,
            )

            def _forward_with(
                pixels,
            ):
                return self.model(
                    input_ids=input_ids,
                    pixel_values=pixels,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True,
                )

            try:

                outputs = _forward_with(
                    pixel_values
                )

            except RuntimeError as e:

                msg = str(e)

                if (
                    "GET was unable to find an engine"
                    in msg
                    or "CUBLAS_STATUS_NOT_INITIALIZED"
                    in msg
                ):

                    print(
                        "FP16 vision forward failed; "
                        "retrying vision input in FP32..."
                    )

                    pixels_f32 = (
                        pixel_values
                        .to(
                            device=self.device,
                            dtype=torch.float32,
                        )
                        .contiguous()
                    )

                    outputs = _forward_with(
                        pixels_f32
                    )

                else:
                    raise

            hidden_states = (
                outputs.hidden_states[-1]
            )

            return hidden_states

        # ---------------------------------------------------------
        # ORIGINAL LLAVA
        # ---------------------------------------------------------

        image_tensors = image_tensors.to(
            self.device
        )

        if (
            image_tensors.device.type != "cpu"
            and image_tensors.dtype
            != torch.float16
        ):
            image_tensors = image_tensors.half()

        def _forward_with(
            image_tensor,
        ):

            return self.model(
                input_ids=input_ids,
                images=image_tensor,
                return_dict=True,
                output_hidden_states=True,
            )

        try:

            images_fwd = image_tensors.contiguous()

            outputs = _forward_with(
                images_fwd
            )

        except RuntimeError as e:

            msg = str(e)

            if (
                "GET was unable to find an engine"
                in msg
                or "CUBLAS_STATUS_NOT_INITIALIZED"
                in msg
            ):

                vision_tower = None

                try:
                    vision_tower = (
                        self.model.get_vision_tower()
                    )
                except Exception:
                    pass

                if vision_tower is not None:

                    vision_tower.to(
                        device=self.device,
                        dtype=torch.float32,
                    )

                images_f32 = (
                    image_tensors
                    .to(
                        device=self.device,
                        dtype=torch.float32,
                    )
                    .contiguous()
                )

                outputs = _forward_with(
                    images_f32
                )

            else:
                raise

        hidden_states = (
            outputs.hidden_states[-1]
        )

        return hidden_states

    # =============================================================
    # FORWARD
    # =============================================================

    def forward(
        self,
        images,
        prompts: list,
    ) -> torch.Tensor:

        return self.extract_hidden_states(
            images,
            prompts,
        )


# =============================================================
# TEST
# =============================================================

def test_extractor():

    print(
        "Testing Frozen LLaVA extractor..."
    )

    llava_base = (
        "checkpoints/llava-1.5-7b-hf"
    )

    llava_lora = (
        "checkpoints/llava-miragehd-prior-bbox"
    )

    extractor = FrozenLLaVAExtractor(
        llava_base_path=llava_base,
        llava_lora_path=llava_lora,
        device="cuda",
    )

    batch_size = 2

    prompts = [
        "How would this RGB scene appear in "
        "long-wave thermal infrared spectrum",

        "Describe the thermal characteristics "
        "of this scene",
    ]

    # IMPORTANT:
    #
    # These tensors are assumed to be ordinary [0,1] RGB
    # images, not already CLIP-normalized tensors.
    #
    dummy_images = torch.rand(
        batch_size,
        3,
        336,
        336,
    )

    hidden_states = extractor(
        dummy_images,
        prompts,
    )

    print(
        f"Hidden states shape: "
        f"{hidden_states.shape}"
    )

    print(
        "Expected final hidden dimension: 4096"
    )

    print(
        "✓ Frozen LLaVA extractor test passed!"
    )

    assert not any(
        p.requires_grad
        for p in extractor.parameters()
    ), "Model should be frozen!"

    print(
        "✓ All parameters frozen"
    )


if __name__ == "__main__":
    test_extractor()
