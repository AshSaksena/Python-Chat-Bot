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
MANIFEST_KEY = "manifest.jsonl"  # Not from Secrets
KB_ID = st.secrets["KB_ID"]
DATA_SOURCE_ID = st.secrets["DATA_SOURCE_ID"]  # From secrets
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
    # Step 1: Textract OCR
    with st.spinner(f"Analyzing {file.name}..."):
        try:
            textract_response = clients['textract'].analyze_document(
                Document={'Bytes': file.getvalue()},
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
                dataSourceId=DATA_SOURCE_ID,
                document={
                    'content': {
                        'text': extracted_text
                    },
                    'metadata': {
                        'filename': file.name,
                        'uploaded_at': str(time.time())
                    }
                }
            )
            # Wait for ingestion to complete
            doc_id = response['document']['documentId']
            if not wait_for_document_ingestion(KB_ID, DATA_SOURCE_ID, doc_id):
                st.error(f"Ingestion timed out or failed for {file.name}.")
                return False
            update_manifest(file.name)
            return True
        except clients['bedrock'].exceptions.ConflictException:
            st.warning(f"{file.name} already exists in KB")
            update_manifest(file.name)
            return True
        except Exception as e:
            st.error(f"Ingestion failed: {str(e)}")
            return False

def wait_for_document_ingestion(kb_id, ds_id, doc_id, timeout=600):
    start = time.time()
    while True:
        try:
            resp = clients['bedrock'].get_knowledge_base_document(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                documentId=doc_id
            )
            status = resp['document']['status']
            if status == "INGESTED":
                return True
            if status in ("FAILED", "DELETED"):
                return False
            if time.time() - start > timeout:
                return False
            time.sleep(5)
        except Exception as e:
            st.error(f"Error checking document ingestion status: {e}")
            return False

# ===== Chatbot Function =====
def get_rag_response(query, session_id):
    prompt_template = f"""You are an Oncology Clinical Specialty ChatBot Assistant.
Use only the information in the knowledge base to answer the question.

Question: {query}

Answer with clinical accuracy. If uncertain, state "I need to consult medical records"."""
    try:
        response = clients['bedrock-agent'].retrieve_and_generate(
            input={'text': query},
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'generationConfiguration': {
                        'promptTemplate': {'textPromptTemplate': prompt_template}
                    },
                    'knowledgeBaseId': KB_ID,
                    'modelArn': MODEL_ARN
                },
                'type': 'KNOWLEDGE_BASE'
            },
            sessionId=session_id
        )
        return response['output']['text'], response.get('sessionId', session_id)
    except Exception as e:
        st.error(f"Clinical error: {str(e)}")
        return "Please consult your physician for immediate concerns.", session_id

# ===== Streamlit UI =====
st.title("Oncology Clinical Specialty ChatBot Assistant 🩺")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

with st.expander("Please Upload Patient Lab Reports and Prescription Notes (PDF)"):
    uploaded_files = st.file_uploader(
        "Upload Patient Lab Reports and Prescription Notes (PDF only)",
        type=['pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        manifest = {entry['filename'] for entry in download_manifest()}
        for file in uploaded_files:
            if file.name in manifest:
                st.success(f"{file.name} already ingested")
                continue
            if process_and_ingest(file):
                st.success(f"Successfully processed {file.name}")

st.divider()

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about patient records..."):
    st.chat_message("user").write(user_input)
    with st.spinner("Consulting medical knowledge..."):
        response, new_session = get_rag_response(
            user_input,
            st.session_state.session_id
        )
        st.session_state.session_id = new_session

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])
