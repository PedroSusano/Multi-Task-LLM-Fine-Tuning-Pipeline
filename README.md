Rental Reviews & Booking - Extraction with Llama 3.2

Sumário
- Estrutura do Projeto
- Objetivo
- Metodologia
- O que foi feito: - Extração de Dados e Análise de Reviews
- Ferramentas e Setup
- Execução
- Conclusões e Aprendizagens
- Extensões Futuras

```text
Estrutura do Projeto
├ Multi-Task LLM Fine-Tuning Pipeline
├──── Edmunds Car Ratings Data Modifications/
│   ├──── train.csv
│   └──── train_fixed.csv
│   ├──── Data Modifications Ed.ipynb
│   └──── Edmunds_Car_Ratings_final.csv
│   └──── Edmunds_Car_Ratings_final.json
├──── Synthetic Booking Emails Modifications/
│   ├──── synthetic_booking_emails_original.csv
│   ├──── synthetic_booking_emails_original.xlsx
│   ├──── Data Modifications Emails.ipynb
│   ├──── synthetic_booking_emails.json
│   └──── synthetic_booking_emails.xlsx
├──── Zero e Few Shot Inference - Reviews.ipynb
├──── Zero Shot Inference Emails.ipynb
├──── Few Shot Inference Emails.ipynb
├──── Zero Shot Inference Emails-Deepseek.ipynb
├──── README.md
├──── Outputs
│   ├──── Edmunds Car Ratings
│   		├──── Edmunds_Car_Ratings_few_shots_checkpoint.json
│   		├──── Edmunds_Car_Ratings_with_predictions_few_shots.csv
│   		├──── Edmunds_Car_Ratings_with_predictions_few_shots.json
│   		├──── Edmunds_Car_Ratings_with_predictions_zero_shot.csv
│   		├──── Edmunds_Car_Ratings_with_predictions_zero_shot.json
│   		├──── metrics_report_few_shots.txt
│   		├──── metrics_report_zero_shot.txt
│   		└──── diferencas_detectadas.csv
│   ├──── Synthetic Booking Emails
│   		├──── synthentic_booking_email_few_shots.json
│   		├──── synthentic_booking_email_zero_shot.json
│   		└──── relatorio_metricas_few_shots_emails.txt
│──── Fine-tuning
│   ├─── Prepara data_py
│   ├─── train.py
│   ├─── pytorch_train.py
│   ├─── batch_inference.ipynb
│   ├─── test2.jsonl
│   ├─── validation2.jsonl
│   ├─── train2.jsonl
│   ├─── predictions.jsonl
│   ├─── relatorio_metricas_fine_tuning_emails.txt
│   ├─── relatorio_metricas_fine_tuning_reviews.txt
│   ├─── Memory Usage.ipynb
│   ├─── relatorio_memoria_estatica_4bit.txt
│   ├─── relatorio_memoria_estatica_16bit.txt
│   ├─── requirements.txt
│   ├─── untitled.txt
│   ├─── Inference Speed.ipynb
│   ├─── Inference Speed.txt
│   ├─── relatorio_memoria_estatica_16bit.txt
│   ├─── training.log
│   ├─── unsloth_compiled_cache
│   	├─── (…)


Ordem de execução:

Data Modifications Ed.ipynb -> Trabalhar os dados do dataset "Edmunds Car Ratings.csv"
↓
Data Modifications Emails.ipynb -> Trabalhar os dados do dataset "synthetic_booking_emails.xlsx"
↓
Zero e Few Shot Inference - Reviews.ipynb -> Fazer inferência em zero shot e few shots para o dataset "Edmunds Car Ratings.csv" e guardar os resultados e as respetivas métricas
↓
Zero Shot Inference Emails.ipynb -> Fazer inferência em zero shot para o dataset "synthetic_booking_emails.xlsx" e guardar os resultados e as respetivas métricas
↓
Few Shot Inference Emails.ipynb -> Fazer inferência em few shots para o dataset "synthetic_booking_emails.xlsx" e guardar os resultados e as respetivas métricas
↓
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here" -> cd finetuning -> nohup python pytorch_train.py > training.log 2>&1 & -> tail -f training.log -> Para treinar o modelo
↓
batch_inference.ipynb -> para fazer inferência no modelo de fine-tuning (para ambos os datasets) e guardar os resultados e as respetivas métricas
↓
Inference Speed -> Para avaliar o tempo de inferência
↓
Memory Usage.ipynb -> Para avaliar o uso de memória
```

Objetivo
Explorar e comparar abordagens de linguagem natural com o modelo LLaMA 3.2 (1B):
- Extração de informação de forma estruturada de emails sintéticos
- Classificação de sentimento e previsão de rating a partir de reviews
- Comparar desempenho de Zero-shot, Few-shot e Fine-Tuning (LoRA)

Metodologia
1. Zero-shot prompting – Modelo responde diretamente sem exemplos
2. Few-shot prompting – Prompts com 2 a 5 exemplos de entrada-saída
3. Avaliação com métricas clássicas (Acuracy, Precision, Recall, F1 Score, MAE, MSE, R²)
4. Fine-tuning com LoRA – Treino supervisionado

O que foi feito
Foram desenvolvidos notebooks com pipelines para extração e análise automática. Utilizámos o endpoint disponibilizado pelo professor no SageMaker da AWS com o modelo LLaMA 3.2 (1B) e testámos os métodos zero-shot e few-shot. Os dados incluem emails sintéticos de reserva de automóveis e reviews e ratings reais.
1.	Extração de Dados dos Emails
Campos extraídos via prompting few-shot:
- Customer Name: 100%
- Car Model: 100%
- Pickup: 98.99%
- Dropoff: 44.24%
- Exact Match: 44.24%

Conclusão: O modelo é confiável para campos explícitos, mas o desempenho deteorou devido ao número máximo de tokens que teve de ser adotado.
2.	Análise de Reviews
Zero-Shot: 
- Acurácia: 73.58%
- F1 Positivo: 0.86
- F1 Neutro: 0.21
- F1 Negativo: 0.39
- RMSE: 1.3378 | R²: -0.3956

Few-Shot: 
- Acurácia: 81.68%
- F1 Positivo: 0.93
- F1 Neutro: 0.06
- F1 Negativo: 0.49
- RMSE: 1.1331 | R²: -0.1827


A abordagem few-shot demonstrou superioridade, mas a classificação de sentimentos neutros continua um desafio. A previsão de rating ainda está limitada sem fine-tuning dedicado.

Ferramentas & Setup
- LLM: LLaMA 3.2 (1B) via AWS SageMaker
- Frameworks: Transformers, PEFT, Accelerate, Evaluate
- Hardware para fine-tuning : ml.g5.16xlarge
- Otimização com uso de KV Cache
- Ambiente Python com requirements.txt disponível

Execução Técnica
O pipeline completo de execução está organizado em duas vertentes principais: Zero-Shot/Few-Shot Prompting e Fine-Tuning Supervisionado com LoRA.

1. Pré-processamento
   - Codificação em utf-8
   - Normalização dos textos (remoção de ruído por ex. colunas e linhas mal formatadas)  

2. Zero-Shot e Few-Shot Prompting  
   - Utilização direta do modelo base com prompts diretos e claros
   - Geração com e sem Chain-of-Thought
   - Análise das respostas extraídas e métricas associadas

3. Fine-Tuning com LoRA  
   - Preparação de datasets (emails e reviews)
   - Treino usando técnicas PEFT (LoRA) recorrendo a Unsloth
   - Inferência via endpoint personalizado AWS e posterior análise dos resultados

4. Inferência e Avaliação  
   - Execução em batch considerando que os prompts são remetidos através do ficheiro com os dados de teste
   - Métricas: Acuracy, Precision, Recall, F1 Score, MAE, MSE, R²), tempo de inferência e memória utilizada
   - Comparação de zero shot, few shots e fine-tuning

5. Resultados e Validação  
   - Verificação com dados reais e sintéticos
   - Validação cruzada com regex para consistência

As expressões regulares (regex) foram fundamentais para extrair informação estruturada dos textos dos emails sintéticos de reservas. Estas técnicas permitem localizar padrões complexos em texto não estruturado, como datas, nomes, modelos de carros e locais de entrega/recolha.
Exemplos de padrões utilizados:
1. Extração de nomes próprios:
    `re.findall(r"(?:Dear|Olá|Hi) (\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", texto)`
2. Datas (formato DD/MM/YYYY ou DD-MM-YYYY):
    `re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", texto)`
3. Modelos de carros com marca e tipo:
    `re.findall(r"\b(?:Toyota|BMW|Volkswagen|Tesla)\s[A-Z][a-zA-Z0-9]+\b", texto)`
4. Locais de pickup e dropoff com preposições:
    `re.findall(r"(?:at|from|to)\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", texto)`
Exemplo de chamada ao endpoint:
python
endpoint_name = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"

Os notebooks contêm prompts estruturados e scripts para comparar as abordagens.


Fine-Tuning Personalizado com LoRA e Unsloth
O processo de fine-tuning foi desenvolvido utilizando o método Low-Rank Adaptation (LoRA), que permite afinar modelos como o LLaMA 3.2 de forma eficiente em recursos computacionais limitados e recorrreu-se à biblioteca Unsloth para a escolha automática dos parâmetros a adaptar.

1. **Preparação dos Dados**:
   - Foram utilizados dois datasets distintos: um com car reviews e outro com emails sintéticos de reserva de carros.
   - O script `Prepare data.py` foi responsável por converter os dados brutos em prompts no formato `chat`, simulando interações com assistentes. Para as reviews, os dados incluíam sentimento, rating e espaço reservado para resposta e plano de ação. Para os emails, os dados foram estruturados no formato JSON.

   - Os dados podem ser formatados e convertidos em string com uma função customizada do template do LLaMA (ex: `<|start_header_id|>user<|end_header_id>`) que se encontra no código ou pode ser utilizado o template transformers do HuggingFace. No nosso caso foi utilizado o template de transformers através de use_transformers_template=True. Os dados foram posteriormente guardados em formato `.jsonl` e divididos em treino, validação e teste (70/15/15).

2. Configuração do Modelo:
   - Foi utilizado o modelo `unsloth/Llama-3.2-1B-Instruct`, carregado com `bfloat16` e `4bit quantization` para reduzir uso de memória.
   - A técnica LoRA foi aplicada sobre os módulos chave: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, com `r=16` e `lora_alpha=32`.
   - A quantidade de parâmetros treináveis foi reduzida para cerca de 1% dos parâmetros totais, mantendo capacidade de adaptação, o que corresponde a aproximadamente 11 milhões de parâmetros.

3. Treino com SFTTrainer:
   - Utilizou-se o `SFTTrainer` da biblioteca `trl`, com estratégia `train_on_responses_only`, focando a adaptação nas respostas do modelo.
   - A métrica de perda (`loss`) foi usada para selecionar o melhor modelo. Foram aplicados `gradient accumulation`, `cosine learning rate schedule`, `warmup`, `8bit AdamW`, e `fp16/bf16` dependendo da GPU.

4. Infraestrutura:
   - O treino decorreu na AWS SageMaker, utilizando instância `ml.g5.16xlarge`.
   - O script `pytorch_train.py` automatizou upload para S3, definiu os valores a utilizar no estimador (batch_size = 16, learning_rate = 2-e4 e max_seq_lenght = 2048) e submeteu o job de treino.
   - O modelo treinado foi guardado em múltiplos formatos: LoRA adapters, merged 16bit e 4bit para uso posterior.

5. Inferência:
 O código está preparado para carregar os adaptadores LoRA, fazer `merge_and_unload` e permitir inferência posterior.

Métricas e Desempenho Pós Fine-Tuning
Resultados no Dataset de Reviews

- Acurácia do sentimento: 87.85%  
- MAE do rating: 0.432  
- MSE do rating: 0.496  
- R² Score: 0.4705

Resultados no Dataset de Emails

- Accuracy em 'Customer Name': 100.00%  
- Accuracy em 'Car Model': 100.00%  
- Accuracy em 'Pickup': 100.00%  
- Accuracy em 'Dropoff': 100.00%  
- Accuracy total (todos os campos corretos): 100.00%

Velocidade de Inferência

Fine-Tuning com KV Cache: 0.250s  
Fine-Tuning sem KV Cache: 0.248s

Uso de Memória Estática
Versão 16-bit
Modelo Base: VRAM 4714.26 MB | RAM 1067.92 MB 
Fine-Tuned: VRAM 4714.26 MB | RAM 1456.64 MB | Disco 2373.61 MB

Versão 4-bit
Modelo Base: VRAM 1023.18 MB | RAM 704.57 MB  
Fine-Tuned: VRAM 1574.37 MB | RAM 713.69 MB | Disco 1067.77 MB

Dificuldades na Inferência
O modelo apresenta dificuldade em distinguir o sentimento neutro do positivo.
Modelo fine-tuned mostrou tendência para prever sempre rating 4 em reviews com palavras ambíguas.

Conclusões e Aprendizagens
- Fine-tuning obteve melhores resultados precedido de few shots
- A integração de exemplos contextuais melhora drasticamente o desempenho.


Extensões Futuras
- Gerar respostas sintéticas e planos de ação (escalations plans) para, pelo menos, uma amostra dos dados de treino, para treinar o modelo também com estes dados e assim este poder gerar respostas e planos de ação.
- Realizar Reinforcement Learning with Human Feedback para adequar as respostas do mesmo
- Criar um front-end (Streamlit ou FastAPI) para demonstração real-time