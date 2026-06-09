import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import os
import boto3
import subprocess
import sys
from datetime import datetime
from huggingface_hub import login

def upload_to_s3(bucket_name: str, local_path: str, s3_prefix: str) -> str:
    """Upload local file to S3 and return S3 URI"""
    try:
        s3_client = boto3.client('s3')
        s3_path = f"{s3_prefix}/{os.path.basename(local_path)}"
        
        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
        s3_client.upload_file(local_path, bucket_name, s3_path)
        return f"s3://{bucket_name}/{s3_path}"
    except Exception as e:
        print(f"❌ Error uploading {local_path} to S3: {e}")
        raise

def run_data_preparation():
    """Run data preparation with error handling"""
    try:
        print("📊 Preparing dataset...")
        
        # Check if prepare_data.py exists
        if not os.path.exists("Prepare data.py"):
            raise FileNotFoundError("Prepare data.py not found in current directory")
        
        # Run with subprocess for better error handling
        result = subprocess.run(
            [sys.executable, "Prepare data.py"],
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Prepare data.py failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Data preparation failed: {result.stderr}")
        
        print("✅ Dataset preparation completed successfully")
        print(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("❌ Data preparation timed out after 5 minutes")
        raise
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        raise

def launch_training():    
    try:
        print("🚀 Starting training job setup...")
        
        # Setup
        role = get_execution_role()
        sagemaker_session = sagemaker.Session()
        bucket_name = sagemaker_session.default_bucket()
        
        print(f"📦 Using S3 bucket: {bucket_name}")
        print(f"🔐 Using IAM role: {role}")
        
        # Login to HuggingFace
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
        
        print("🤗 Logging into HuggingFace...")
        login(token=hf_token)
        print("✅ HuggingFace login successful")
        
        # Prepare dataset
        #run_data_preparation() #--> only run for the first time.
        
        # Check if data files exist
        required_files = ["train2.jsonl", "test2.jsonl", "validation2.jsonl"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required data file not found: {file}")
        
        # Upload data to S3
        print("📤 Uploading data to S3...")
        train_s3_uri = upload_to_s3(bucket_name, "train2.jsonl", "nlp-genai-grupo5/data")
        test_s3_uri = upload_to_s3(bucket_name, "test2.jsonl", "nlp-genai-grupo5/data")
        val_s3_uri = upload_to_s3(bucket_name, "validation2.jsonl", "nlp-genai-grupo5/data")
        
        print(f"✅ Train data uploaded: {train_s3_uri}")
        print(f"✅ Test data uploaded: {test_s3_uri}")
        print(f"✅ Validation data uploaded: {val_s3_uri}")
        
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("⚠️  requirements.txt not found, creating basic one...")
            with open("requirements.txt", "w") as f:
                f.write("unsloth\n")
        
        # Check if train.py exists
        if not os.path.exists("train.py"):
            raise FileNotFoundError("train.py not found in current directory")
        
        # Create job name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        job_name = f"grupo5-fine-tuning-pytorch-{timestamp}"
        
        # Define environment variables
        environment = {
            'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
            'HUGGINGFACE_HUB_TOKEN': hf_token,  # Pass token to training environment
        }
        
        print("🏗️  Creating PyTorch estimator...")
        
        # Define the PyTorch estimator
        pytorch_estimator = PyTorch(
            entry_point='train.py',
            source_dir='.',
            role=role,
            framework_version='2.6.0',
            py_version='py312',
            instance_count=1,
            instance_type='ml.g5.16xlarge',
            hyperparameters={
                'epochs': 3,
                'batch_size': 16,  # Use underscore, not hyphen
                'learning_rate': 2e-4,  # Use underscore, not hyphen
                'max_seq_length': 2048,  # Use underscore, not hyphen
                'test_path': test_s3_uri
            },
            environment=environment,
            output_path=f's3://{bucket_name}/nlp-genai-grupo5/output',
            dependencies=['requirements.txt']
        )
        
        # Define input channels
        input_channels = {
            'train': train_s3_uri,
            'test': test_s3_uri,
            'validation': val_s3_uri
        }
        
        print(f"🎯 Starting training job: {job_name}")
        print(f"💾 Input channels: {input_channels}")
        
        # Start training
        pytorch_estimator.fit(inputs=input_channels,job_name=job_name)
        
        print("✅ Training job submitted successfully!")
        return pytorch_estimator
        
    except Exception as e:
        print(f"❌ Error in launch_training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':  # Fixed the syntax error
    try:
        estimator = launch_training()
    except Exception as e:
        print(f"💥 Script failed: {e}")
        sys.exit(1)