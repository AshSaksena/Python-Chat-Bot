import boto3
import streamlit as st
import os
import json
from botocore.exceptions import ClientError

# ===== Get AWS Credentials from Secrets Manager =====
def get_aws_secret(secret_name, region_name):
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        st.error(f"Could not retrieve secret: {e}")
        return None

    # Parse and return the secret string as a dict
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    else:
        st.error("Secret is in binary format, which is not supported here.")
        return None

# Set these to your secret name and region
secret_name = "myAWSCreds"
region_name = "us-east-2"

# Retrieve secrets
aws_creds = get_aws_secret(secret_name, region_name)

if aws_creds:
    # Set credentials as environment variables (so boto3 uses them)
    os.environ["AWS_ACCESS_KEY_ID"] = aws_creds["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_creds["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = aws_creds.get("AWS_DEFAULT_REGION", region_name)
else:
    st.error("Failed to load AWS credentials from Secrets Manager.")
    st.stop()  # Stop execution if credentials are not loaded

# Get KB_ID and MODEL_ARN directly from the secret
KB_ID = aws_creds.get("KB_ID")
MODEL_ARN = aws_creds.get("MODEL_ARN")

if not KB_ID or not MODEL_ARN:
    st.error("KB_ID or MODEL_ARN not found in Secrets Manager! Please update your secret.")
    st.stop()

# ===== Initialize Bedrock Client (cached) =====
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=region_name
    )

client = get_bedrock_client()

# ===== RAG Response Function =====
def get_rag_response(query, session_id=""):
    prompt_template = """\
Use this context to answer the question. If unsure, say "I don't know".

Context:
$search_results$

Question:
$query$

Answer comprehensively. Cite sources if available.
Answer:"""
    try:
        params = {
            'input': {'text': query},
            'retrieveAndGenerateConfiguration': {
                'knowledgeBaseConfiguration': {
                    'generationConfiguration': {
                        'promptTemplate': {'textPromptTemplate': prompt_template}
                    },
                    'knowledgeBaseId': KB_ID,
                    'modelArn': MODEL_ARN
                },
                'type': 'KNOWLEDGE_BASE'
            }
        }
        if session_id:
            params['sessionId'] = session_id

        response = client.retrieve_and_generate(**params)
        return response['output']['text'], response.get('sessionId', "")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Sorry, I encountered an error.", session_id

# ===== Streamlit UI =====
st.title("Amazon Bedrock RAG Chatbot 🚀")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = ""

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask about your knowledge base...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Generating response..."):
        response, new_session_id = get_rag_response(user_input, st.session_state.session_id)
        st.session_state.session_id = new_session_id

    with st.chat_message("assistant"):
        st.write(response)

    # Append to history
    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])
