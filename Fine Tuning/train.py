from unsloth import FastLanguageModel, is_bfloat16_supported
#import re
import json
import torch
#import numpy as np
import argparse
from transformers import TrainingArguments, DataCollatorForSeq2Seq 
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--test_path', type=str, required=True)

    return parser.parse_known_args()


def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


'''
Não foi possivel avaliar o modelo através do rating_mse ou da accuracy do sentimento pois consome demasiada memória, além disso não foi possível fazer inferência nos dados de testes e gerar o predictions

def compute_metrics(eval_preds, tokenizer, eval_dataset):
    predictions, labels = eval_preds
    pred_texts = []
    for pred in predictions:
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        try:
            pred_texts.append(pred_text.split("<|start_header_id|>assistant<|end_header_id>")[1].split("<|eot_id>")[0].strip())
        except IndexError:
            pred_texts.append("")  # Lida com outputs incorretos
    label_texts = []
    for label in labels:
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        try:
            label_texts.append(label_text.split("<|start_header_id|>assistant<|end_header_id>")[1].split("<|eot_id>")[0].strip())
        except IndexError:
            label_texts.append("")  # Lida com outputs incorretos

    car_rating_metrics = {"sentiment_accuracy": [], "rating_mse": []}
    email_extraction_metrics = {"field_accuracy": []}

    sentiment_pattern = r"\*\*Sentiment\*\*: (Positive|Neutral|Negative)"
    rating_pattern = r"\*\*Rating\*\*: (\d+\.\d{1,3})"
    json_pattern = r"```json\n([\s\S]*?)\n```"

    for pred_text, label_text, example in zip(pred_texts, label_texts, eval_dataset):
        task = example["task"]

        if task == "car_review_rating":
            pred_sentiment = re.search(sentiment_pattern, pred_text)
            pred_rating = re.search(rating_pattern, pred_text)
            true_sentiment = re.search(sentiment_pattern, label_text)
            true_rating = re.search(rating_pattern, label_text)

            if pred_sentiment and pred_rating and true_sentiment and true_rating:
                car_rating_metrics["sentiment_accuracy"].append(
                    pred_sentiment.group(1) == true_sentiment.group(1)
                )
                car_rating_metrics["rating_mse"].append(
                    (float(pred_rating.group(1)) - float(true_rating.group(1))) ** 2
                )

        elif task == "email_extraction":
            pred_json_match = re.search(json_pattern, pred_text)
            true_json_match = re.search(json_pattern, label_text)

            if pred_json_match and true_json_match:
                try:
                    pred_data = json.loads(pred_json_match.group(1))
                    true_data = json.loads(true_json_match.group(1))
                    field_accuracy = sum(pred_data.get(k, "") == true_data.get(k, "") 
                                        for k in true_data) / len(true_data)
                    email_extraction_metrics["field_accuracy"].append(field_accuracy)
                except json.JSONDecodeError:
                    email_extraction_metrics["field_accuracy"].append(0.0)
    
    return {
        "car_review_rating_sentiment_accuracy": np.mean(car_rating_metrics["sentiment_accuracy"]) if car_rating_metrics["sentiment_accuracy"] else 0.0,
        "car_review_rating_rating_mse": np.mean(car_rating_metrics["rating_mse"]) if car_rating_metrics["rating_mse"] else float("inf"),
        "email_extraction_field_accuracy": np.mean(email_extraction_metrics["field_accuracy"]) if email_extraction_metrics["field_accuracy"] else 0.0
    }


def process_test_set(trainer, test_dataset, tokenizer, output_file="/opt/ml/output/predictions.jsonl"):
    print("🚀 Processing test set: generating predictions and evaluating metrics...")
    
    # Generate predictions using trainer.predict
    predictions = trainer.predict(test_dataset)
    pred_texts = []
    for pred in predictions.predictions:
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        try:
            pred_texts.append(pred_text.split("<|start_header_id|>assistant<|end_header_id>")[1].split("<|eot_id>")[0].strip())
        except IndexError:
            pred_texts.append("")  # Handle malformed output
    label_texts = []
    for label in predictions.label_ids:
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        try:
            label_texts.append(label_text.split("<|start_header_id|>assistant<|end_header_id>")[1].split("<|eot_id>")[0].strip())
        except IndexError:
            label_texts.append("")  # Handle malformed label
    
    # Save predictions
    predictions_to_save = []
    for example, pred_text in zip(test_dataset, pred_texts):
        task = example["task"]
        input_text = example["text"]
        user_prompt = input_text.split("<|start_header_id|>assistant<|end_header_id>")[0]
        predictions_to_save.append({"task": task, "input": user_prompt, "prediction": pred_text})
    
    with open(output_file, "w") as f:
        for pred in predictions_to_save:
            f.write(json.dumps(pred) + "\n")
    print(f"✅ Predictions saved to {output_file}")

    # Compute metrics
    eval_results = compute_metrics((predictions.predictions, predictions.label_ids, test_dataset), tokenizer, test_dataset)
    eval_results = {f"test_{k}": v for k, v in eval_results.items()}
    print(f"Test set metrics: {eval_results}")
    return eval_results
'''


if __name__ == '__main__':
    args, _ = parse_args()

    # Load model and tokenizer
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Count the total number of parameters in the original model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the original model: {total_params:,}")

    # Aplicar o LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Count the number of trainable parameters after applying LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after applying LoRA: {trainable_params:,}")
    
    # Calculate the percentage of trainable parameters compared to the original model
    percentage = (trainable_params / total_params) * 100
    print(f"Percentage of trainable parameters: {percentage:.2f}%")
    
    # Load training data
    train_dataset = load_jsonl_data('/opt/ml/input/data/train/train2.jsonl')
    # Reduce to 20 rows
    # train_dataset = train_dataset.select(range(20))
    
    val_dataset = load_jsonl_data('/opt/ml/input/data/validation/validation2.jsonl')
    test_dataset = load_jsonl_data('/opt/ml/input/data/test/test2.jsonl')
    
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",                   # Directory to save model checkpoints and outputs
        num_train_epochs=args.epochs,                 # Number of training epochs
        per_device_train_batch_size=args.batch_size,  # Batch size per device during training
        gradient_accumulation_steps=4,                # Number of steps to accumulate gradients before updating
        learning_rate=args.learning_rate,             # Initial learning rate
        lr_scheduler_type="cosine",                   # Learning rate scheduler type
        warmup_ratio=0.1,                             # Proportion of training steps for learning rate warmup
        logging_steps=10,                             # Frequency of logging loss and metrics
        optim="adamw_8bit",                           # Optimizer type
        eval_strategy="epoch",                  # Evaluation strategy to use during training
        save_strategy="epoch",                        # Save strategy to use during training
        save_total_limit=1,                           # Limit the total number of checkpoints
        fp16=not is_bfloat16_supported(),             # Use 16-bit (mixed) precision if bfloat16 is not supported
        bf16=is_bfloat16_supported(),                 # Use bfloat16 precision if supported
        load_best_model_at_end=True,                  # Load the best model at the end of training
        #metric_for_best_model="car_review_sentiment_accuracy",           # Metric to use for selecting the best model -> Sentimento
        #greater_is_better=True,                      # Quanto menor melhor
        metric_for_best_model="loss",
        weight_decay=0.01,                            # Weight decay to apply (if any)
        seed=3407                                     # Random seed for reproducibility
    )


    # Initialize trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        #compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, val_dataset), # Utilizamos o dataset de validação para escolher o melhor modelo com maior acerto no sentimento
        max_seq_length = args.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )

    from unsloth.chat_templates import train_on_responses_only
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

   
    # Train the model
    
    # Treinar
    trainer.train()
    
    # Save the model
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")


    # Realiza previsões no dataset de teste
    #process_test_set(trainer, test_dataset, tokenizer) -> Teve de ser comentado pois consumia demasiada memória fazer a inferência nos dados de teste

    # Saving the adapters 
    # Save only the LoRA adapters
    model.save_pretrained("/opt/ml/model/lora_adapter", save_adapter=True)

    def load_model_with_lora(model_id, lora_adapter_path, max_seq_length, dtype, load_in_4bit, adapter_name="default"):
        # Load the base model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length= max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=load_in_4bit,
        )
    
        # Define the LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
        )
    
        # Apply the LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        # Load the LoRA adapter weights
        model.load_adapter(lora_adapter_path, adapter_name=adapter_name)

        # Merge the LoRA weights into the base model
        # Aqui seria -> model = model.merge_and_unload() . linha 186

        return model, tokenizer

    # Define paths and parameters
    # model_id = "your_model_id"  # Replace with your model ID
    lora_adapter_path = "/opt/ml/model/lora_adapter"
    max_seq_length = args.max_seq_length  # Set your desired max sequence length
    dtype = torch.bfloat16
    load_in_4bit = True
    
    # List of save configurations (folder names match config.yaml model_format values)
    save_configs = [
        {"path": "/opt/ml/model/merged_16bit", "method": "merged_16bit"},
        {"path": "/opt/ml/model/merged_4bit", "method": "merged_4bit_forced"},
        # {"path": "/opt/ml/model/gguf_8bit", "method": "gguf", "quantization": ""},
        # {"path": "/opt/ml/model/gguf_16bit", "method": "gguf", "quantization": "f16"},
        # {"path": "/opt/ml/model/gguf_q4km", "method": "gguf", "quantization": "q4_k_m"}
    ]
    
    for config in save_configs:
        try:
            # Determine load_in_4bit based on save method
            use_4bit = "4bit" in config["method"]

            # Reload the base model with the LoRA adapters
            model, tokenizer = load_model_with_lora(
                model_id, lora_adapter_path, max_seq_length, dtype, use_4bit
            )
    
            # Perform the save operation based on the method
            if config["method"] == "gguf":
                model.save_pretrained_gguf(
                    config["path"], tokenizer, quantization_method=config["quantization"]
                )
            else:
                model.save_pretrained_merged(
                    config["path"], tokenizer, save_method=config["method"]
                )
    
            print(f"Model saved successfully at {config['path']}")
        except Exception as e:
            print(f"Failed to save model at {config['path']}: {str(e)}")