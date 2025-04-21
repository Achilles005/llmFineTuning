#!/usr/bin/env python3
import os
import re
from typing import Dict, Any, Tuple, List

import pandas as pd
import torch
from abc import ABC, abstractmethod
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, PPOTrainer, PPOConfig
from fastapi import FastAPI, HTTPException
import docker

# ---------- Common Utilities ----------
class DataPreprocessor:
    @staticmethod
    def normalize_citations(text: str) -> str:
        text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '[Citation]', text)
        text = re.sub(r'\([A-Z][a-zA-Z]+(?: et al\.)?,?\s*\d{4}\)', '[Citation]', text)
        text = re.sub(r'[A-Z][a-zA-Z]+(?: et al\.)?\s+\(\s*\d{4}\s*\)', '[Citation]', text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def extract_title_abstract(text: str) -> Tuple[str, str]:
        title_match = re.search(r'\bTitle:\s*(.*)', text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        abstract_match = re.search(r'\bAbstract:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else text.strip()
        return title, abstract

class DatasetManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_dataset(self) -> pd.DataFrame:
        data_records = []
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                label = "Noncancer" if "non-cancer" in folder.lower() else "Cancer"
                for file in os.listdir(folder_path):
                    if file.endswith('.txt'):
                        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                            text = DataPreprocessor.normalize_citations(f.read())
                            _, abstract = DataPreprocessor.extract_title_abstract(text)
                            data_records.append({"text": abstract, "label": label})
        return pd.DataFrame(data_records)

# ---------- Base Trainer Class ----------
class BaseTrainer(ABC):
    def __init__(self, model_name: str, dataset: pd.DataFrame):
        self.model_name = model_name
        self.dataset = dataset
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def configure_quantization(self) -> BitsAndBytesConfig:
        """Return quantization config for bitsandbytes."""
        pass

    @abstractmethod
    def train(self):
        """Fine-tune the model."""
        pass

    @abstractmethod
    def save_model(self, output_dir: str):
        """Save model and tokenizer to output_dir."""
        pass

    def _prepare_data(self, train_size: float = 0.8) -> Tuple[Dataset, Dataset]:
        df = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        train_end = int(train_size * len(df))
        train_df, eval_df = df[:train_end], df[train_end:]
        return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

# ---------- SFT Trainer ----------
class SFTrainer(BaseTrainer):
    def configure_quantization(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

    def train(self):
        # Load and prepare model
        bnb_config = self.configure_quantization()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.config.use_cache = False

        # Prepare data
        train_ds, eval_ds = self._prepare_data()
        def format_fn(example):
            enc = self.tokenizer(
                f"Text: {example['text']} Label: {example['label']}",
                return_tensors="pt",
                padding="longest"
            )
            return {"input_ids": enc.input_ids[0], "attention_mask": enc.attention_mask[0]}
        train_ds = train_ds.map(format_fn, remove_columns=["text","label"])
        eval_ds = eval_ds.map(format_fn, remove_columns=["text","label"])

        # Training args
        training_args = TrainingArguments(
            output_dir="sft_output",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds
        )
        trainer.train()

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# ---------- GRPO Trainer ----------
class GRPOTrainer(BaseTrainer):
    def configure_quantization(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )

    def train(self):
        # Load and prepare model
        bnb_config = self.configure_quantization()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.config.use_cache = False

        # Prepare data
        train_ds, _ = self._prepare_data()

        # PPO configuration
        ppo_config = PPOConfig(
            model_name=self.model_name,
            learning_rate=1e-5,
            batch_size=2,
            forward_batch_size=1,
            optimize_cuda_cache=True
        )
        ppo_trainer = PPOTrainer(model=self.model, tokenizer=self.tokenizer, **ppo_config.__dict__)

        # Simple reward function
        def compute_reward(prompt: str, response: str, label: str) -> float:
            match = re.search(r'C\w+', response)
            pred = match.group(0).capitalize() if match else "None"
            return 1.0 if pred == label else 0.0

        for example in train_ds:
            prompt = f"Text: {example['text']}\n<reasoning></reasoning><answer>"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=5)
            response = self.tokenizer.decode(out[0], skip_special_tokens=True)
            reward = compute_reward(prompt, response, example["label"])
            ppo_trainer.step([prompt], [response], [reward])

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# ---------- Evaluation & API & Docker (unchanged) ----------
class ModelEvaluator:
    def evaluate(self, model: BaseTrainer, df: pd.DataFrame) -> Dict[str, Any]:
        # Stub for evaluating saved model
        raise NotImplementedError

class FineTuningAPI:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/start-training")
        async def start_training(config: Dict[str, Any]):
            raise HTTPException(status_code=501, detail="Not implemented")

        @self.app.get("/model-status")
        async def get_status(model_id: str):
            raise HTTPException(status_code=501, detail="Not implemented")

        @self.app.post("/deploy")
        async def deploy_model(model_path: str):
            raise HTTPException(status_code=501, detail="Not implemented")

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()

    def build_image(self, model_path: str, tag: str = "llm-service"):
        dockerfile = f"""
        FROM python:3.9
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY {model_path} /app/model
        COPY llm_service.py /app/
        CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        return self.client.images.build(fileobj=dockerfile.encode(), tag=tag, rm=True)

if __name__ == "__main__":
    dm = DatasetManager("/path/to/dataset")
    df = dm.load_dataset()
    sft = SFTrainer("meta-llama/Llama-3-8B", df)
    sft.train()
    sft.save_model("./sft_model")
    grpo = GRPOTrainer("microsoft/phi-2", df)
    grpo.train()
    grpo.save_model("./grpo_model")
