import os
import torch
import json
import random
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset, load_dataset

# Import OpenAI client
import openai
from tqdm import tqdm
from enum import Enum
import time

# --- Configuration ---

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN", None)
if HF_TOKEN is None:
    print(
        "Warning: Hugging Face token not found. Set HUGGING_FACE_TOKEN environment variable."
    )

# Get Fireworks API Key
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", None)
if FIREWORKS_API_KEY is None:
    raise ValueError(
        "Fireworks API key required. Set FIREWORKS_API_KEY environment variable."
    )


class JudgeModel(Enum):
    # Use Fireworks AI model identifiers
    LLAMA4_SCOUT_FW = "accounts/fireworks/models/llama4-scout-instruct-basic"

    @classmethod
    def get_all_models(cls) -> List[str]:
        return [model.value for model in cls]


# --- Unlearning Pipeline ---


class UnlearningRLPipeline:
    """
    Implements an unlearning pipeline using PPO and an LLM-as-a-judge for reward signaling.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        judge_model_enum: JudgeModel = JudgeModel.LLAMA4_SCOUT_FW,
        fireworks_api_key: str = FIREWORKS_API_KEY,  # Use key from top
        hf_token: Optional[str] = HF_TOKEN,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        judge_batch_size: int = 4,
        judge_api_retries: int = 3,
        judge_api_delay: float = 1.0,
        default_unlearning_criteria: str = "The response should not contain harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. It should also avoid generating personally identifiable information about non-public figures.",
    ):
        print(f"Initializing pipeline on device: {device}")
        self.device = device
        self.judge_model_name = judge_model_enum.value
        self.judge_batch_size = judge_batch_size
        self.judge_api_retries = judge_api_retries
        self.judge_api_delay = judge_api_delay
        self.default_unlearning_criteria = default_unlearning_criteria
        self.hf_token = hf_token

        if not self.hf_token:
            print(
                "Warning: Hugging Face token is missing. Model loading might fail for gated models."
            )

        # Enable TF32
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print("Enabling TF32")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Model and Tokenizer (using hf_token)
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
        )
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left", token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- OpenAI SDK Client pointing to Fireworks AI ---
        print(
            f"Initializing OpenAI SDK client for Fireworks AI judge: {self.judge_model_name}"
        )
        self.client = openai.OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=fireworks_api_key,  # Use the key passed or from env var
        )
        print("OpenAI SDK client configured for Fireworks AI.")

    # _format_prompt method remains the same

    def get_llm_judge_rewards(
        self, prompts: List[str], responses: List[str], criteria: List[str]
    ) -> List[float]:
        """
        Gets rewards from the LLM judge (via Fireworks AI using OpenAI SDK).
        Enforces JSON object response format. Retries API calls on failure.
        """
        rewards = []
        num_batches = (
            len(prompts) + self.judge_batch_size - 1
        ) // self.judge_batch_size

        print(
            f"Getting judge rewards for {len(prompts)} samples in {num_batches} batches via Fireworks AI (Judge: {self.judge_model_name})..."
        )

        judge_system_prompt = """You are a precise response evaluator. Your task is to determine if an AI assistant's response violates specific unlearning criteria, given a user prompt and the assistant's response. Focus ONLY on whether the unlearning criteria are violated. Respond ONLY with a JSON object containing 'reasoning' (a brief explanation) and 'should_unlearn' (boolean: true if criteria are VIOLATED, false otherwise).

JSON FORMAT:
{
    "reasoning": "Brief explanation here.",
    "should_unlearn": boolean
}"""

        for i in tqdm(
            range(num_batches), desc=f"Judge Batches ({self.judge_model_name})"
        ):
            batch_start = i * self.judge_batch_size
            batch_end = min((i + 1) * self.judge_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_responses = responses[batch_start:batch_end]
            batch_criteria = criteria[batch_start:batch_end]

            api_results = []  # Store results for the batch
            for idx_in_batch, (prompt, response, crit) in enumerate(
                zip(batch_prompts, batch_responses, batch_criteria)
            ):
                user_message = f"""Evaluate the following AI response based ONLY on the unlearning criteria provided.

**User Prompt:**
{prompt}

**AI Response:**
{response}

**Unlearning Criteria:**
{crit}

Respond ONLY in the specified JSON format."""

                completion = None
                for attempt in range(self.judge_api_retries):
                    try:
                        # Use the OpenAI SDK client instance
                        completion = self.client.chat.completions.create(
                            model=self.judge_model_name,  # Model identifier for Fireworks
                            messages=[
                                {"role": "system", "content": judge_system_prompt},
                                {"role": "user", "content": user_message},
                            ],
                            response_format={
                                "type": "json_object"
                            },  # Request JSON output
                            temperature=0.0,
                            max_tokens=150,  # Limit output tokens
                        )
                        api_results.append(completion)  # Store successful result
                        break  # Exit retry loop on success
                    except Exception as e:
                        # Catch potential OpenAI API errors or other exceptions
                        print(
                            f"Warning: Judge API call failed (Attempt {attempt + 1}/{self.judge_api_retries}) for sample {batch_start + idx_in_batch}. Error: {e}"
                        )
                        if attempt < self.judge_api_retries - 1:
                            print(f"Retrying in {self.judge_api_delay} seconds...")
                            time.sleep(self.judge_api_delay)
                        else:
                            print("Max retries reached. Assigning neutral reward.")
                            api_results.append(None)  # Mark as failed

            # Process results for the batch
            for result in api_results:
                if result is None:  # Handle API call failure after retries
                    rewards.append(0.0)
                    continue
                try:
                    # Access message content correctly via OpenAI SDK structure
                    content = result.choices[0].message.content
                    if content is None:
                        print(
                            f"Warning: Judge API returned None content. Assigning neutral reward."
                        )
                        rewards.append(0.0)
                        continue
                    content = content.strip()

                    if not content.startswith("{") or not content.endswith("}"):
                        print(
                            f"Warning: Judge output doesn't look like JSON: '{content}'. Assigning neutral reward."
                        )
                        rewards.append(0.0)
                        continue

                    judge_output = json.loads(content)
                    should_unlearn = judge_output.get("should_unlearn")

                    if not isinstance(should_unlearn, bool):
                        print(
                            f"Warning: 'should_unlearn' key missing or not boolean in judge output: {judge_output}. Assigning neutral reward."
                        )
                        rewards.append(0.0)
                        continue

                    reward = 1.0 if not should_unlearn else -1.0
                    rewards.append(reward)

                except json.JSONDecodeError as e:
                    print(
                        f"Error decoding judge JSON ({self.judge_model_name}): {e}\nRaw content: '{content}'"
                    )
                    rewards.append(0.0)
                except Exception as e:
                    # Catch potential errors accessing result attributes
                    print(
                        f"Error processing judge reward ({self.judge_model_name}): {e}"
                    )
                    rewards.append(0.0)

        print(f"Finished getting {len(rewards)} judge rewards.")
        return rewards

    # prepare_dataset method remains the same

    def train(
        self,
        dataset: Dataset,
        ppo_config: PPOConfig,
        generation_kwargs: Dict,
        output_dir: str = "ppo_unlearned_model",
    ):
        """Train the model using PPOTrainer."""
        if "query" not in dataset.features:
            raise ValueError("Dataset must contain a 'query' column for PPOTrainer")

        print("Loading reference model...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token,
        )
        print("Reference model loaded.")

        print("Initializing PPOTrainer...")
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            data_collator=None,
        )
        print("PPOTrainer initialized.")

        print("Starting PPO training loop...")
        # --- Training Loop (unchanged logic, relies on updated get_llm_judge_rewards) ---
        for epoch in range(ppo_config.ppo_epochs):
            print(f"--- PPO Epoch {epoch + 1}/{ppo_config.ppo_epochs} ---")
            epoch_rewards = []
            for batch in tqdm(
                ppo_trainer.dataloader, desc=f"Epoch {epoch + 1} Batches"
            ):
                query_tensors = batch["input_ids"].to(self.device)

                response_tensors_list = []
                response_texts_list = []
                with torch.no_grad():
                    for query_tensor in query_tensors:
                        if query_tensor.dim() == 0:
                            continue
                        current_generation_kwargs = generation_kwargs.copy()
                        response = ppo_trainer.model.generate(
                            input_ids=query_tensor.unsqueeze(0),
                            **current_generation_kwargs,
                        )
                        response_tensors_list.append(response.squeeze(0))
                        prompt_len = query_tensor.shape[0]
                        response_text = self.tokenizer.decode(
                            response.squeeze(0)[prompt_len:], skip_special_tokens=True
                        )
                        response_texts_list.append(response_text)

                query_texts = self.tokenizer.batch_decode(
                    query_tensors, skip_special_tokens=True
                )
                batch_criteria = batch.get(
                    "unlearning_criteria",
                    [self.default_unlearning_criteria] * len(query_texts),
                )

                if len(response_texts_list) != len(query_texts):
                    print(
                        f"Warning: Mismatch in query ({len(query_texts)}) and response ({len(response_texts_list)}) counts. Skipping batch."
                    )
                    continue

                rewards = self.get_llm_judge_rewards(
                    query_texts, response_texts_list, batch_criteria
                )
                reward_tensors = [
                    torch.tensor(r, device=self.device, dtype=torch.float32)
                    for r in rewards
                ]
                epoch_rewards.extend(rewards)

                try:
                    if not (
                        len(query_tensors)
                        == len(response_tensors_list)
                        == len(reward_tensors)
                    ):
                        raise ValueError(
                            f"Batch size mismatch before PPO step: Q={len(query_tensors)}, R(list)={len(response_tensors_list)}, Rew={len(reward_tensors)}"
                        )

                    stats = ppo_trainer.step(
                        list(query_tensors), response_tensors_list, reward_tensors
                    )
                    # Optional: Print basic stats if desired
                    # print(f"Step stats obj: {stats}") # Logged by TRL internally usually

                except Exception as e:
                    print(f"\nError during ppo_trainer.step: {e}")
                    print(
                        f"Sizes: Queries={len(query_tensors)}, Responses List={len(response_tensors_list)}, Rewards={len(reward_tensors)}"
                    )
                    print("Skipping faulty batch.")
                    continue

            avg_epoch_reward = (
                sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            )
            print(
                f"--- End Epoch {epoch + 1}: Average Reward = {avg_epoch_reward:.4f} ---"
            )

            checkpoint_dir = f"{output_dir}/epoch_{epoch + 1}"
            print(f"Saving model checkpoint to {checkpoint_dir}")
            ppo_trainer.save_pretrained(checkpoint_dir)

        print("Training finished.")
        final_dir = f"{output_dir}/final"
        print(f"Saving final model to {final_dir}")
        ppo_trainer.save_pretrained(final_dir)
        print(f"Final model saved to: {final_dir}")


# --- Main Execution ---


def main(
    model_name: str = "meta-llama/Llama-3.2-1B",
    judge_model_str: str = JudgeModel.LLAMA4_SCOUT_FW.value,
    dataset_name: str = "microsoft/toxigen",
    dataset_split: str = "train",
    num_samples: int = 100,
    output_dir: str = "ppo_unlearned_model_1b_scout",
    # PPO Config Args
    learning_rate: float = 1.41e-6,
    ppo_epochs: int = 4,
    batch_size: int = 16,
    mini_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    kl_penalty: float = 0.1,
    seed: int = 42,
    # Generation Config Args
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
):
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Removed wandb initialization

    # --- PPO Configuration ---
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ppo_epochs=ppo_epochs,
        kl_penalty="kl",
        target_kl=kl_penalty,
        seed=seed,
        # Removed log_with config
        optimize_cuda_cache=True,
    )

    # --- Generation Configuration ---
    temp_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    pad_token_id = (
        temp_tokenizer.eos_token_id
        if temp_tokenizer.pad_token_id is None
        else temp_tokenizer.pad_token_id
    )
    del temp_tokenizer

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
    }

    # --- Initialize Pipeline ---
    try:
        judge_model_enum = JudgeModel(judge_model_str)
    except ValueError:
        print(
            f"Error: Invalid judge_model '{judge_model_str}'. Available Fireworks: {[m.value for m in JudgeModel]}"
        )
        return

    # Ensure FIREWORKS_API_KEY is set before this point
    pipeline = UnlearningRLPipeline(
        model_name=model_name,
        judge_model_enum=judge_model_enum,
        hf_token=HF_TOKEN,
        # API key is read from env var by default now
    )

    # --- Prepare Data ---
    dataset = pipeline.prepare_dataset(
        dataset_name=dataset_name,
        split=dataset_split,
        num_samples=num_samples,
    )

    # --- Train ---
    pipeline.train(
        dataset=dataset,
        ppo_config=ppo_config,
        generation_kwargs=generation_kwargs,
        output_dir=output_dir,
    )

    # Removed wandb.finish()
    print("Pipeline execution finished.")


if __name__ == "__main__":
    main(
        model_name="meta-llama/Llama-3.2-1B",
        judge_model_str=JudgeModel.LLAMA4_SCOUT_FW.value,
        dataset_name="microsoft/toxigen",
        num_samples=50,  # Keep very small for initial testing
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        output_dir="llama32_1b_unlearned_scout_test_openai_sdk",  # Updated dir name
    )
