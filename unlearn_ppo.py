import os
import torch
import json
import random
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)  # Added BitsAndBytesConfig for optional 4-bit
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    setup_chat_format,
)
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

# Import OpenAI client SDK
import openai
from tqdm import tqdm
from enum import Enum
import time
import traceback  # For detailed error printing
import numpy  # For seeding

# --- Configuration ---

# Load environment variables from .env file if it exists
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", None)
if FIREWORKS_API_KEY is None:
    raise ValueError(
        "Fireworks API key required. Set FIREWORKS_API_KEY environment variable."
    )


class JudgeModel(Enum):
    """Enumeration for supported judge models via Fireworks AI."""

    LLAMA4_SCOUT_FW = "accounts/fireworks/models/llama4-scout-instruct-basic"

    @classmethod
    def get_all_models(cls) -> List[str]:
        return [model.value for model in cls]


# --- Unlearning Pipeline ---


class UnlearningRLPipeline:
    """
    Implements an unlearning pipeline using PPO and an LLM-as-a-judge
    (via Fireworks AI/OpenAI SDK) for reward signaling.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        judge_model_enum: JudgeModel = JudgeModel.LLAMA4_SCOUT_FW,
        fireworks_api_key: str = FIREWORKS_API_KEY,
        hf_token: Optional[str] = HF_TOKEN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        judge_batch_size: int = 4,
        judge_api_retries: int = 3,
        judge_api_delay: float = 1.0,
        default_unlearning_criteria: str = (
            "The response should not contain harmful, unethical, racist, sexist, "
            "toxic, dangerous, or illegal content. It should also avoid generating "
            "personally identifiable information about non-public figures."
        ),
        use_4bit: bool = False,  # Option to load model in 4-bit for memory saving
    ):
        """Initializes the Unlearning Pipeline."""
        print(f"Initializing pipeline on device: {device}")
        self.device = device
        self.model_name = model_name
        self.judge_model_name = judge_model_enum.value
        self.judge_batch_size = judge_batch_size
        self.judge_api_retries = judge_api_retries
        self.judge_api_delay = judge_api_delay
        self.default_unlearning_criteria = default_unlearning_criteria
        self.hf_token = hf_token
        self.use_4bit = use_4bit

        if not self.hf_token and ("meta-llama" in model_name or "Llama" in model_name):
            print(
                f"Warning: Hugging Face token (HF_TOKEN) is missing but model '{model_name}' likely requires it."
            )

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print("Enabling TF32 optimization for Ampere+ GPU.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # --- Quantization Config (Optional) ---
        quantization_config = None
        if self.use_4bit:
            if not torch.cuda.is_available():
                print(
                    "Warning: 4-bit quantization requested but no CUDA device found. Loading in default precision."
                )
                self.use_4bit = False  # Disable if no CUDA
            else:
                print("Using 4-bit quantization (BitsAndBytes).")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Or float16 depending on GPU
                )

        # --- Load Model and Tokenizer ---
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": torch.bfloat16
            if not self.use_4bit
            else None,  # dtype handled by quantization_config if 4bit
            "device_map": "auto",
            "trust_remote_code": True,
            "token": self.hf_token,
            "quantization_config": quantization_config,
        }
        print(f"Loading model with ValueHead: {self.model_name}")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(**model_kwargs)

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",  # Essential for PPO/ValueHead
            token=self.hf_token,
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                print(
                    f"Setting tokenizer pad_token to eos_token ({self.tokenizer.eos_token})."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                print(
                    "Warning: Tokenizer lacks EOS and PAD tokens. Adding PAD token '[PAD]'."
                )
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # Important: Resize model embeddings to accommodate the new token
                self.model.resize_token_embeddings(len(self.tokenizer))
                # If using a ref model later, it might also need resizing or reloading.

        # TRL recommendation for chat models, may adapt prompt format
        # setup_chat_format(self.model, self.tokenizer)

        # --- Initialize OpenAI SDK Client for Fireworks AI ---
        print(
            f"Initializing OpenAI SDK client for Fireworks AI judge: {self.judge_model_name}"
        )
        self.client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=fireworks_api_key,
        )
        print("OpenAI SDK client configured for Fireworks AI.")

    def get_llm_judge_rewards(
        self, prompts: List[str], responses: List[str], criteria: List[str]
    ) -> List[float]:
        """Gets rewards from the LLM judge via Fireworks AI."""
        rewards = []
        num_prompts = len(prompts)
        if num_prompts == 0:
            return []
        num_batches = (num_prompts + self.judge_batch_size - 1) // self.judge_batch_size

        print(
            f"Getting judge rewards for {num_prompts} samples in {num_batches} batches (Judge: {self.judge_model_name})..."
        )

        judge_system_prompt = """You are a precise response evaluator. Your task is to determine if an AI assistant's response violates specific unlearning criteria, given a user prompt and the assistant's response. Focus ONLY on whether the unlearning criteria are violated. Respond ONLY with a JSON object containing 'reasoning' (a brief explanation) and 'should_unlearn' (boolean: true if criteria are VIOLATED, false otherwise). JSON FORMAT: {"reasoning": "Brief explanation here.", "should_unlearn": boolean}"""

        for i in tqdm(
            range(num_batches), desc=f"Judge Batches ({self.judge_model_name})"
        ):
            batch_start, batch_end = (
                i * self.judge_batch_size,
                min((i + 1) * self.judge_batch_size, num_prompts),
            )
            batch_prompts, batch_responses, batch_criteria = (
                prompts[batch_start:batch_end],
                responses[batch_start:batch_end],
                criteria[batch_start:batch_end],
            )

            api_results = []
            for idx_in_batch, (prompt, response, crit) in enumerate(
                zip(batch_prompts, batch_responses, batch_criteria)
            ):
                user_message = f"""Evaluate the following AI response based ONLY on the unlearning criteria provided.\n\n**User Prompt:**\n{prompt}\n\n**AI Response:**\n{response}\n\n**Unlearning Criteria:**\n{crit}\n\nRespond ONLY in the specified JSON format."""
                completion = None
                for attempt in range(self.judge_api_retries):
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.judge_model_name,
                            messages=[
                                {"role": "system", "content": judge_system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=150,
                        )
                        api_results.append(completion)
                        break
                    except openai.RateLimitError as rle:
                        wait_time = self.judge_api_delay * (2**attempt)
                        print(
                            f"\nRate limit error from Fireworks AI ({rle}). Waiting {wait_time:.2f}s (Attempt {attempt + 1}/{self.judge_api_retries})..."
                        )
                        time.sleep(wait_time)
                        if attempt == self.judge_api_retries - 1:
                            print(
                                "Max retries reached after rate limit. Assigning neutral reward (0.0)."
                            )
                            api_results.append(None)
                    except Exception as e:
                        print(
                            f"\nWarning: Judge API call failed (Attempt {attempt + 1}/{self.judge_api_retries}) sample {batch_start + idx_in_batch}. Error: {type(e).__name__} - {e}"
                        )
                        if attempt < self.judge_api_retries - 1:
                            time.sleep(self.judge_api_delay)
                        else:
                            print(
                                "Max retries reached. Assigning neutral reward (0.0)."
                            )
                            api_results.append(None)

            for result in api_results:
                reward = 0.0  # Default neutral reward
                if result is not None:
                    try:
                        message = result.choices[0].message
                        content = message.content.strip() if message.content else ""
                        if content.startswith("{") and content.endswith("}"):
                            judge_output = json.loads(content)
                            should_unlearn = judge_output.get("should_unlearn")
                            if isinstance(should_unlearn, bool):
                                reward = 1.0 if not should_unlearn else -1.0
                                # reasoning = judge_output.get('reasoning', 'N/A')
                                # if reward == -1.0: print(f"\n[REWARD -1.0] Judge: {reasoning}")
                            else:
                                print(
                                    f"\nWarning: Judge JSON missing/invalid 'should_unlearn': {judge_output}"
                                )
                        else:
                            print(
                                f"\nWarning: Judge output not valid JSON: '{content}'"
                            )
                    except json.JSONDecodeError as e:
                        print(f"\nError decoding judge JSON: {e}\nRaw: '{content}'")
                    except Exception as e:
                        print(
                            f"\nError processing judge reward: {type(e).__name__} - {e}"
                        )
                rewards.append(reward)

        print(f"Finished getting {len(rewards)} judge rewards.")
        if len(rewards) != num_prompts:
            print(
                f"Warning: Reward count mismatch ({len(rewards)} vs {num_prompts}). Padding with 0.0."
            )
            rewards.extend([0.0] * (num_prompts - len(rewards)))
        return rewards

    def prepare_dataset(
        self, dataset_name: str, split: str = "train", num_samples: Optional[int] = None
    ) -> Dataset:
        """Loads and prepares the dataset for PPO training."""
        print(f"Loading dataset: {dataset_name} [{split}]")
        dataset = load_dataset(dataset_name, split=split, token=self.hf_token)

        if num_samples is not None and 0 < num_samples < len(dataset):
            print(f"Selecting {num_samples} random samples.")
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        elif num_samples is not None:
            print(
                f"Requested {num_samples}, dataset has {len(dataset)}. Using all samples."
            )

        # Adapt ToxiGen specifically
        if dataset_name == "microsoft/toxigen":
            print(
                "Adapting 'microsoft/toxigen': using 'text' as 'prompt', removing other columns."
            )
            if "text" not in dataset.features:
                raise ValueError("ToxiGen needs 'text' column.")
            dataset = dataset.rename_column("text", "prompt")
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col != "prompt"]
            )
        elif "prompt" not in dataset.features:
            raise ValueError(
                f"Dataset needs 'prompt' column or specific handling. Found: {dataset.column_names}"
            )

        # Add necessary columns and tokenize
        def process_example(example):
            example["unlearning_criteria"] = self.default_unlearning_criteria
            # 'query' is the raw text prompt used by PPO internally
            example["query"] = (
                str(example["prompt"]) if example["prompt"] is not None else ""
            )
            # 'input_ids' are the tokenized query, needed for generation
            tokenized = self.tokenizer(
                example["query"], truncation=True, max_length=512, padding=False
            )  # Max length added
            example["input_ids"] = tokenized["input_ids"]
            # example["attention_mask"] = tokenized["attention_mask"] # Not strictly needed by PPOTrainer if collator handles it
            return example

        dataset = dataset.map(process_example)

        # Filter out empty prompts/queries after processing
        initial_len = len(dataset)
        dataset = dataset.filter(
            lambda x: len(x["query"]) > 0 and len(x["input_ids"]) > 0
        )
        if len(dataset) < initial_len:
            print(f"Filtered {initial_len - len(dataset)} empty samples.")

        # Keep only columns needed by PPOTrainer and our generation loop
        # PPOTrainer needs 'query' (string) and 'input_ids' (tokens) from the dataset.
        # We also need 'unlearning_criteria'.
        final_columns = ["query", "input_ids", "unlearning_criteria"]
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in final_columns]
        )

        print(f"Prepared dataset with {len(dataset)} samples.")
        print("Dataset features:", dataset.features)
        if len(dataset) > 0:
            print("Example entry:\n", dataset[0])
        else:
            print("Dataset is empty after preparation.")
        return dataset

    def train(
        self,
        dataset: Dataset,
        ppo_config: PPOConfig,
        generation_kwargs: Dict,
        output_dir: str = "ppo_unlearned_model",
    ):
        """Trains the model using PPOTrainer."""
        required_cols = {"query", "input_ids", "unlearning_criteria"}
        if not required_cols.issubset(dataset.features):
            raise ValueError(
                f"Dataset must contain: {required_cols}. Found: {list(dataset.features.keys())}"
            )
        if len(dataset) == 0:
            print("Error: Cannot train on an empty dataset.")
            return

        # --- Load Reference Model ---
        print(f"Loading reference model: {self.model_name}")
        ref_model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": torch.bfloat16 if not self.use_4bit else None,
            "device_map": "auto",  # Can be on different device
            "trust_remote_code": True,
            "token": self.hf_token,
            "quantization_config": None,  # Ref model usually not quantized or needs separate handling
            # Load without value head if memory is tight, PPOTrainer can handle it
            # But AutoModelForCausalLMWithValueHead is safer / standard practice
        }
        # If main model was quantized, ref model should ideally match precision but *without* value head initially if using below approach
        # Load ref model also with Value Head for simplicity and consistency here
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            **ref_model_kwargs
        )
        # If main model's embeddings were resized, resize ref model too
        # if self.tokenizer.pad_token == '[PAD]': # Check if we added a token
        #    ref_model.resize_token_embeddings(len(self.tokenizer))
        print("Reference model loaded.")

        # --- Initialize PPOTrainer ---
        print("Initializing PPOTrainer...")
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            data_collator=None,  # Use default collator
        )
        print("PPOTrainer initialized.")

        # --- PPO Training Loop ---
        print("Starting PPO training loop...")
        num_experience_epochs = 1  # Single pass over dataset for experience generation
        print(f"Running {num_experience_epochs} experience generation pass(es).")

        # Update generation kwargs with correct token IDs from loaded tokenizer
        if generation_kwargs.get("pad_token_id") is None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if generation_kwargs.get("eos_token_id") is None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        print("Using generation kwargs:", generation_kwargs)

        for epoch in range(num_experience_epochs):
            print(
                f"--- Experience Generation Pass {epoch + 1}/{num_experience_epochs} ---"
            )
            epoch_rewards_list = []
            for batch in tqdm(ppo_trainer.dataloader, desc=f"Pass {epoch + 1} Batches"):
                query_tensors = batch[
                    "input_ids"
                ]  # These are List[List[int]], collator makes them tensor
                query_texts = batch["query"]  # List[str]

                # Ensure query_tensors is a tensor on the correct device (collator should handle this)
                if isinstance(query_tensors, list):  # Fallback if collator didn't batch
                    query_tensors = self.tokenizer.pad(
                        {"input_ids": query_tensors}, return_tensors="pt"
                    )["input_ids"].to(self.device)
                elif isinstance(query_tensors, torch.Tensor):
                    query_tensors = query_tensors.to(self.device)
                else:
                    print(
                        f"Warning: Unexpected type for query_tensors: {type(query_tensors)}. Skipping batch."
                    )
                    continue

                response_tensors_list = []  # List[Tensor] for PPO step
                response_texts_list = []  # List[str] for judge

                # --- Generation ---
                # Using generate directly on the batch tensor is much faster
                try:
                    with torch.no_grad():
                        # Ensure generation kwargs are valid
                        valid_gen_kwargs = generation_kwargs.copy()
                        if valid_gen_kwargs.get("top_k") == 0:
                            valid_gen_kwargs.pop("top_k", None)
                        if (
                            valid_gen_kwargs.get("top_p") is None
                            or valid_gen_kwargs.get("top_p") >= 1.0
                        ):
                            valid_gen_kwargs.pop("top_p", None)

                        response_tensors = ppo_trainer.generate(
                            query_tensors,
                            return_prompt=False,  # Keep only generated tokens for decoding text
                            **valid_gen_kwargs,
                        )
                        # response_tensors contain only the *new* tokens here

                        # Decode generated part for the judge
                        response_texts_list = self.tokenizer.batch_decode(
                            response_tensors, skip_special_tokens=True
                        )

                        # For PPOTrainer.step, we need the full sequence (prompt + response)
                        # We can reconstruct this manually or use PPOTrainer's generate slightly differently if needed
                        # Let's reconstruct:
                        full_response_tensors_list = []
                        for i in range(len(query_tensors)):
                            # Find actual length of prompt (ignoring padding)
                            # Assuming left padding from tokenizer
                            prompt_end_index = (
                                torch.max(
                                    torch.where(
                                        query_tensors[i] != self.tokenizer.pad_token_id,
                                        torch.arange(
                                            query_tensors[i].size(0), device=self.device
                                        ),
                                        torch.tensor(-1, device=self.device),
                                    )
                                )
                                + 1
                            )
                            prompt = query_tensors[i][:prompt_end_index]

                            # Find actual length of generated response (ignoring padding)
                            if response_tensors[i].numel() > 0:
                                gen_end_index = (
                                    torch.max(
                                        torch.where(
                                            response_tensors[i]
                                            != self.tokenizer.pad_token_id,
                                            torch.arange(
                                                response_tensors[i].size(0),
                                                device=self.device,
                                            ),
                                            torch.tensor(-1, device=self.device),
                                        )
                                    )
                                    + 1
                                )
                                response = response_tensors[i][:gen_end_index]
                            else:  # Handle empty generation case
                                response = torch.tensor(
                                    [], dtype=torch.long, device=self.device
                                )

                            full_response = torch.cat((prompt, response), dim=0)
                            full_response_tensors_list.append(full_response)

                        response_tensors_list = full_response_tensors_list  # Use the reconstructed full tensors

                except Exception as gen_e:
                    print(f"\nError during batch generation: {gen_e}")
                    traceback.print_exc()
                    # Cannot proceed with this batch
                    continue

                # --- Get Rewards ---
                batch_criteria = batch["unlearning_criteria"]  # List[str]
                if not (
                    len(query_texts)
                    == len(response_texts_list)
                    == len(response_tensors_list)
                    == len(batch_criteria)
                ):
                    print(
                        f"\nCRITICAL WARNING: Mismatch after generation/reconstruction. "
                        f"Q:{len(query_texts)}, R_txt:{len(response_texts_list)}, R_tens:{len(response_tensors_list)}, C:{len(batch_criteria)}. Skipping."
                    )
                    continue

                rewards = self.get_llm_judge_rewards(
                    query_texts, response_texts_list, batch_criteria
                )
                try:
                    # Ensure rewards are scalar tensors
                    reward_tensors = [
                        torch.tensor(r, device=self.device, dtype=torch.float32)
                        for r in rewards
                    ]
                    # Check they are indeed scalar
                    if any(rt.ndim != 0 for rt in reward_tensors):
                        raise ValueError(
                            f"Reward tensors must be scalar. Got shapes: {[rt.shape for rt in reward_tensors]}"
                        )
                except Exception as rew_e:
                    print(
                        f"\nError preparing reward tensors: {rew_e}. Rewards: {rewards}. Skipping batch."
                    )
                    continue

                epoch_rewards_list.extend(rewards)

                # --- PPO Step ---
                try:
                    # PPOTrainer needs List[Tensor] for queries, List[Tensor] for responses, List[Tensor] for rewards
                    query_tensors_list = [
                        q for q in query_tensors
                    ]  # Convert batch tensor to list of tensors

                    stats = ppo_trainer.step(
                        query_tensors_list, response_tensors_list, reward_tensors
                    )
                    log_batch = {"query": query_texts, "response": response_texts_list}
                    ppo_trainer.log_stats(
                        stats, log_batch, rewards
                    )  # Log float rewards

                except Exception as ppo_e:
                    print(
                        f"\nCRITICAL ERROR during ppo_trainer.step: {type(ppo_e).__name__} - {ppo_e}"
                    )
                    print(
                        f"Batch sizes - Q:{len(query_tensors_list)}, R:{len(response_tensors_list)}, Rew:{len(reward_tensors)}"
                    )
                    # Print shapes for debugging
                    # for i in range(min(5, len(query_tensors_list))): # Print first few shapes
                    #    print(f"  Sample {i}: Q shape={query_tensors_list[i].shape}, R shape={response_tensors_list[i].shape}, Rew shape={reward_tensors[i].shape}")
                    traceback.print_exc()
                    print("Skipping faulty batch.")
                    continue  # Skip this batch

            # --- End of Pass Logging & Checkpointing ---
            avg_epoch_reward = (
                sum(epoch_rewards_list) / len(epoch_rewards_list)
                if epoch_rewards_list
                else 0
            )
            print(
                f"--- End Pass {epoch + 1}: Average Reward = {avg_epoch_reward:.4f} ---"
            )

            if output_dir:
                checkpoint_dir = os.path.join(output_dir, f"pass_{epoch + 1}")
                print(f"Saving checkpoint to {checkpoint_dir}")
                try:
                    ppo_trainer.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                except Exception as save_e:
                    print(f"\nWarning: Failed to save checkpoint: {save_e}")

        # --- End of Training ---
        print("PPO Training finished.")
        if output_dir:
            final_dir = os.path.join(output_dir, "final")
            print(f"Saving final model to {final_dir}")
            try:
                ppo_trainer.save_pretrained(final_dir)
                self.tokenizer.save_pretrained(final_dir)
                print(f"Final model saved to: {final_dir}")
            except Exception as save_e:
                print(f"\nWarning: Failed to save final model: {save_e}")
        else:
            print("output_dir not specified, skipping final model save.")


# --- Main Execution Block ---


def main(
    # Model & Judge
    model_name: str = "meta-llama/Llama-3.2-1B",
    judge_model_str: str = JudgeModel.LLAMA4_SCOUT_FW.value,
    use_4bit: bool = False,  # Set to True to try 4-bit loading
    # Dataset
    dataset_name: str = "microsoft/toxigen",
    dataset_split: str = "train",
    num_samples: int = 100,
    # Output
    output_dir: str = "ppo_unlearned_model_1b_scout_openai_sdk",
    # PPO Hyperparameters
    learning_rate: float = 1.41e-6,
    ppo_epochs: int = 4,  # <<< CORRECTED: Optimization epochs per batch (used by PPOConfig) >>>
    batch_size: int = 16,  # Experience collection batch size
    mini_batch_size: int = 4,  # Optimization mini-batch size
    gradient_accumulation_steps: int = 1,
    adap_kl_ctrl: bool = True,
    init_kl_coef: float = 0.1,
    target_kl: float = 0.1,
    clip_range: float = 0.2,
    vf_coef: float = 0.1,
    seed: int = 42,
    # Generation Hyperparameters
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
):
    """Main function to configure and run the unlearning pipeline."""
    random.seed(seed)
    torch.manual_seed(seed)
    np_seed = random.randint(0, 2**32 - 1)
    numpy.random.seed(np_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed} (Torch) and {np_seed} (NumPy)")

    # --- PPO Config ---
    if mini_batch_size <= 0:
        raise ValueError("mini_batch_size > 0 required")
    actual_grad_accum = gradient_accumulation_steps
    if mini_batch_size * gradient_accumulation_steps > batch_size:
        actual_grad_accum = max(1, batch_size // mini_batch_size)
        print(
            f"Warning: mini_batch_size*grad_accum > batch_size. "
            f"Using grad_accum={actual_grad_accum}."
        )
    else:
        actual_grad_accum = gradient_accumulation_steps

    ppo_config = PPOConfig(
        learning_rate=learning_rate,  # base LR
        batch_size=batch_size,  # total experience batch size
        mini_batch_size=mini_batch_size,  # per-step optimization minibatch
        gradient_accumulation_steps=actual_grad_accum,  # accumulated steps
        num_ppo_epochs=ppo_epochs,  # renamed from ppo_epochs
        kl_coef=init_kl_coef,  # fixed KL coefficient (replaces kl_penalty/target)
        cliprange=clip_range,  # renamed from clip_range
        vf_coef=vf_coef,  # value-function loss coefficient
        # the following parameters use their default values:
        # whiten_rewards (False), kl_estimator ("k1"),
        # cliprange_value (same as cliprange), gamma (1.0), lam (0.95),
        # ds3_gather_for_generation (True)
    )
    print("--- PPO Configuration ---")
    config_dict = ppo_config.to_dict()
    for key in sorted(config_dict.keys()):
        print(f"{key}: {config_dict[key]}")
    print("-------------------------")

    # --- Generation Config ---
    # Token IDs will be fetched inside pipeline based on loaded tokenizer
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,  # Keep 0 if disabling
        "top_p": top_p,  # Keep 1.0 if disabling
        "do_sample": do_sample,
        "pad_token_id": None,  # Will be set by pipeline
        "eos_token_id": None,  # Will be set by pipeline
    }
    print("--- Generation Configuration (Initial) ---")
    print(generation_kwargs)
    print("-----------------------------------------")

    # --- Initialize & Run Pipeline ---
    try:
        judge_model_enum = JudgeModel(judge_model_str)
    except ValueError:
        print(
            f"Error: Invalid judge_model '{judge_model_str}'. Available: {[m.value for m in JudgeModel]}"
        )
        return

    if not FIREWORKS_API_KEY:
        print("Error: FIREWORKS_API_KEY missing.")
        return

    pipeline = UnlearningRLPipeline(
        model_name=model_name,
        judge_model_enum=judge_model_enum,
        fireworks_api_key=FIREWORKS_API_KEY,
        hf_token=HF_TOKEN,
        use_4bit=use_4bit,  # Pass 4-bit flag
    )

    try:
        dataset = pipeline.prepare_dataset(
            dataset_name=dataset_name,
            split=dataset_split,
            num_samples=num_samples,
        )
        if len(dataset) == 0:
            print("Error: Dataset empty after preparation.")
            return
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        traceback.print_exc()
        return

    try:
        print(f"\n--- Starting Training ---")
        pipeline.train(
            dataset=dataset,
            ppo_config=ppo_config,
            generation_kwargs=generation_kwargs,
            output_dir=output_dir,
        )
        print(f"--- Training Completed ---")
    except Exception as e:
        print(f"\nError during training: {type(e).__name__} - {e}")
        traceback.print_exc()

    print("Pipeline execution finished.")


if __name__ == "__main__":
    print("Starting main execution...")
    main(
        model_name="meta-llama/Llama-3.2-1B",
        judge_model_str=JudgeModel.LLAMA4_SCOUT_FW.value,
        use_4bit=False,  # Set True if you have GPU memory constraints and want to try 4-bit
        dataset_name="microsoft/toxigen",
        dataset_split="train",
        num_samples=50,  # Debugging value
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        ppo_epochs=2,  # Debugging value (Number of optimization epochs per collected batch)
        learning_rate=1.41e-6,
        adap_kl_ctrl=True,
        init_kl_coef=0.1,
        target_kl=0.1,
        max_new_tokens=60,  # Debugging value
        output_dir="llama32_1b_unlearned_scout_test_final_fix3",
    )
    print("Main execution finished.")
