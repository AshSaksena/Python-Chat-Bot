import streamlit as st
import boto3
import uuid
import json
import time
from botocore.exceptions import ClientError, BotoCoreError

# ===== AWS Configuration =====
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
S3_BUCKET = st.secrets["S3_BUCKET"]  # For manifest only
MANIFEST_KEY = "manifest.jsonl"  # Hardcoded manifest file name
KB_ID = st.secrets["KB_ID"]
DATA_SOURCE_ID = st.secrets["DATA_SOURCE_ID"]
MODEL_ARN = st.secrets["MODEL_ARN"]

# ===== Initialize AWS Clients =====
@st.cache_resource
def get_aws_clients():
    return {
        'textract': boto3.client('textract', region_name=AWS_REGION,
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        'bedrock-agent': boto3.client('bedrock-agent-runtime', region_name=AWS_REGION,
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        'bedrock': boto3.client('bedrock', region_name=AWS_REGION,
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY),
        's3': boto3.client('s3', region_name=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    }

clients = get_aws_clients()

# ===== Manifest Management =====
def download_manifest():
    try:
        obj = clients['s3'].get_object(Bucket=S3_BUCKET, Key=MANIFEST_KEY)
        return [json.loads(line) for line in obj['Body'].read().decode('utf-8').splitlines()]
    except ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            return []
        st.error(f"Error downloading manifest: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error downloading manifest: {e}")
        return []

def update_manifest(filename):
    try:
        manifest = download_manifest()
        manifest.append({
            "filename": filename,
            "ingested_at": str(time.time()),
            "status": "INGESTED"
        })
        clients['s3'].put_object(
            Bucket=S3_BUCKET,
            Key=MANIFEST_KEY,
            Body="\n".join(json.dumps(entry) for entry in manifest).encode('utf-8')
        )
    except Exception as e:
        st.error(f"Error updating manifest: {e}")

# ===== Direct Ingestion Pipeline =====
def process_and_ingest(file):
    # ===== Pre-validation Before Textract Call =====
    if file.type != "application/pdf":
        st.error("Please upload a PDF file.")
        return False

    file_bytes = file.getvalue()
    if len(file_bytes) == 0:
        st.error("Uploaded file is empty.")
        return False

    if len(file_bytes) > 5 * 1024 * 1024:
        st.error("PDF is larger than 5MB. Please upload a smaller file.")
        return False

    # Step 1: Textract OCR
    with st.spinner(f"Analyzing {file.name}..."):
        try:
            textract_response = clients['textract'].analyze_document(
                Document={'Bytes': file_bytes},
                FeatureTypes=['FORMS', 'TABLES', 'HANDWRITING']
            )
            extracted_text = '\n'.join(
                block['Text'] for block in textract_response['Blocks'] 
                if block['BlockType'] in ['LINE', 'WORD']
            )
        except Exception as e:
            st.error(f"OCR failed: {str(e)}")
            return False

    # Step 2: Direct Bedrock ingestion
    with st.spinner(f"Ingesting {file.name} into Bedrock Knowledge Base..."):
        try:
            response = clients['bedrock'].create_knowledge_base_document(
                knowledgeBaseId=KB_ID,
                data
