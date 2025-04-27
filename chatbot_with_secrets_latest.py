import streamlit as st
import boto3

# ===== Get AWS Credentials from Streamlit Secrets =====
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]
KB_ID = st.secrets["KB_ID"]
MODEL_ARN = st.secrets["MODEL_ARN"]

# ===== Initialize Bedrock Client (cached) =====
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=AWS_DEFAULT_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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
