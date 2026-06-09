import pandas as pd
import json
import os
import re
from typing import List, Dict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

def create_car_rating_messages(review: str, sentiment: str = None, rating: float = None) -> List[Dict]:
    """
    Prompt no formato de mensagens para fine-tuning no dataset Edmunds_Car_Ratings.
    """
    # System message
    system_message = (
        "You are an expert automotive assistant specialized in analyzing car reviews. "
        "Your tasks are to: "
        "1. Classify the sentiment of the review as Positive, Neutral, or Negative. "
        "2. Predict the numerical rating (1.000 to 5.000, where 1.000 is very negative and 5.000 is very positive). "
        "3. If based on the review you consider it necessary, generate a professional, empathetic response to the review, addressing the reviewer's concerns or praise. "
        "4. If the review indicates issues (e.g., Negative sentiment or low rating), provide an escalation plan "
        "to address the concerns, including specific actions for customer service or technical teams. "
        "Ensure your responses are concise, professional, and tailored to the review content."
    )

    # User Message
    user_message = (
        "Please analyze the following car review and provide: "
        "1. The sentiment (Positive, Neutral, or Negative). "
        "2. The predicted rating (1.000 to 5.000). "
        "3. If based on the review you consider it necessary, a professional and empathetic response to the reviewer. "
        "4. An escalation plan if applicable, based on the review content.\n\n"
        f"Review: {review}"
    )

    # Separou-se do assistant para podermos fazer inferência em outros dados ou outros datasets onde apenas esteja a review (se no futuro quisermos)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Assistant message
    # Apenas incluimos sentimento e rating para treino pois não possuímos respostas ou planos de ação
    # Mas na inferência o modelo vai gerar todas as componentes
    if sentiment and rating is not None:
        assistant_message = (
            f"**Sentiment**: {sentiment}\n"
            f"**Rating**: {rating:.3f}\n"
            "**Response**: [To be generated during inference, if necessary]\n"
            "**Escalation Plan**: [To be generated during inference, if applicable]"
        )
        messages.append({"role": "assistant", "content": assistant_message})
    return messages


def create_email_extraction_messages(email_body: str, extracted_data: dict = None) -> List[Dict]:
    """Prompt no formato de mensagens para fine-tuning no dataset Synthetic_booking_emails"""
    system_message = (
        "You are an AI assistant specialized in extracting structured information from emails. "
        "Your task is to extract the following fields from the email body and return them in JSON format: "
        "customer_name, car_model, pickup_date_location, dropoff_date_location."
    )
    user_message = (
        "Please extract the following fields from the email in JSON format: "
        "customer_name, car_model, pickup_date_location, dropoff_date_location.\n\n"
        f"Email: {email_body}"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    if extracted_data and any(extracted_data.values()): # apenas vai incluir no assistant àquelas que têm pelo menos um campo preenchido
        assistant_message = f"```json\n{json.dumps(extracted_data, indent=2)}\n```"
        messages.append({"role": "assistant", "content": assistant_message})
    return messages


def format_template(messages: List[Dict], use_transformers_template: bool = False, tokenizer=None) -> str:
    """Formatar as mensagens usando ou o custom Template para o Llama 3.2 ou o template de transformers."""
    if use_transformers_template:
        if tokenizer is None:
            raise ValueError("Tokenizer tem de ser fornecido quando use_transformers_template=True")
        return tokenizer.apply_chat_template(messages, tokenize=False).strip()
    else:
        formatted = ""
        if any(msg["role"] == "system" for msg in messages):
            system_msg = next(msg for msg in messages if msg["role"] == "system")
            formatted += f"<|start_header_id|>system<|end_header_id>\n{system_msg['content']}\n<|eot_id>|\n"
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"<|start_header_id|>user<|end_header_id>\n{msg['content']}\n<|eot_id>|\n"
            elif msg["role"] == "assistant" and msg["role"] != "system":
                formatted += f"<|start_header_id|>assistant<|end_header_id>\n{msg['content']}\n<|eot_id>|\n"
        return formatted.strip()


def extract_email_fields(email_body: str) -> dict:
    """Regex para extrair a informação dos emails de forma estruturada."""
    extracted_data = {
        "customer_name": "",
        "car_model": "",
        "pickup_date_location": "",
        "dropoff_date_location": ""
    }
    name_pattern = r"(?:Dear|Caro\(a\)|Olá|Hello)\s+([A-Za-z\s]+?)(?:,|\n|$)"
    car_model_pattern = r"(?:Car|Vehicle|Viatura):\s*([A-Za-z0-9\s]+?)(?:\n|$)"
    pickup_pattern = r"(?:Pick-up|Levantar|Pick-up date):\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?:\s+\([^)]+\))?)\s*(?:at|em|\()\s*([A-Za-z\s]+?)(?:\n|$|\))"
    dropoff_pattern = r"(?:Return|Drop-off|Devolver|Drop-off date):\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?:\s+\([^)]+\))?)\s*(?:at|em|\()\s*([A-Za-z\s]+?)(?:\n|$|\))"

    name_match = re.search(name_pattern, email_body, re.IGNORECASE)
    if name_match:
        extracted_data["customer_name"] = name_match.group(1).strip()
    car_model_match = re.search(car_model_pattern, email_body, re.IGNORECASE)
    if car_model_match:
        extracted_data["car_model"] = car_model_match.group(1).strip()
    pickup_match = re.search(pickup_pattern, email_body, re.IGNORECASE)
    if pickup_match:
        date_time, location = pickup_match.groups()
        extracted_data["pickup_date_location"] = f"{date_time.strip()} at {location.strip()}"
    dropoff_match = re.search(dropoff_pattern, email_body, re.IGNORECASE)
    if dropoff_match:
        date_time, location = dropoff_match.groups()
        extracted_data["dropoff_date_location"] = f"{date_time.strip()} at {location.strip()}"

    return extracted_data


def prepare_data(
    ratings_path="/home/sagemaker-user/NLPGENAI-GRUPO_10/Grupo 5/Edmunds Car Ratings Data Modifications/Edmunds_Car_Ratings_final.csv",
    email_path="/home/sagemaker-user/NLPGENAI-GRUPO_10/Grupo 5/Synthetic Booking Emails Modifications/synthetic_booking_emails.json",
    output_dir=".",
    use_transformers_template: bool = False
):
    """Preparar os datasets para análise de reviews e extração de emails."""
    # Iniciar o tokenizer se se usar o transformers template
    tokenizer = None
    if use_transformers_template:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    
    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    # Process car ratings dataset
    if os.path.exists(ratings_path):
        print("📊 Processing car reviews dataset...")
        df = pd.read_csv(ratings_path)
        for _, row in df.iterrows():
            messages = create_car_rating_messages(row["Review"], row["Real_Label"], row["Rating"])
            all_data.append({
                "task": "car_review_rating",
                "text": format_template(messages, use_transformers_template, tokenizer)
            })

    # Process email dataset
    if os.path.exists(email_path):
        print("📧 Processing email dataset...")
        with open(email_path, "r") as f:
            email_data = json.load(f)
        for email in email_data:
            email_body = email["body"]
            extracted_data = extract_email_fields(email_body)
            messages = create_email_extraction_messages(email_body, extracted_data)
            all_data.append({
                "task": "email_extraction",
                "text": format_template(messages, use_transformers_template, tokenizer)
            })


    # Dividir o dataset em treino, validação e teste
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)  # 70% treino, 30% temporário para divisão seguinte
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 15% validação, 15% teste conforme solicitado pelo professor
    
    # Guardar em JSONL
    for split, data in [("train2", train_data), ("validation2", val_data), ("test2", test_data)]:
        with open(os.path.join(output_dir, f"{split}.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Saved {split}.jsonl with {len(data)} examples.")
        

if __name__ == "__main__":
    prepare_data(
        ratings_path="/home/sagemaker-user/NLPGENAI-GRUPO_10/Grupo 5/Edmunds Car Ratings Data Modifications/Edmunds_Car_Ratings_final.csv",
        email_path="/home/sagemaker-user/NLPGENAI-GRUPO_10/Grupo 5/Synthetic Booking Emails Modifications/synthetic_booking_emails.json",
        use_transformers_template=True  # Enable transformers template by default
    )