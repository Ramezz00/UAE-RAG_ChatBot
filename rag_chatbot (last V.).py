!pip install openai pandas streamlit
!pip install scikit-learn
!pip install pyngrok

from openai import AzureOpenAI
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = AzureOpenAI(
    api_key="BqDcDlWbo6D5UYktjh9PAFCHEHAOEh271ALbVTIAzVmz0DD9eNUDJQQJ99BAACHrzpqXJ3w3AAABACOGqLmb",
    api_version="2024-10-21",
    azure_endpoint="https://aldar-test.openai.azure.com/"
)

df = pd.read_csv('/content/uae_real_estate_2024.csv')
df['description'] = df['description'].fillna("")
df = df[df['description'].str.len() > 0]

def generate_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return np.zeros(1536)
        return embedding
    except Exception as e:
        return np.zeros(1536)

df['embedding'] = df['description'].apply(lambda x: generate_embedding(x))

def find_relevant_properties(query, top_k=5):
    query_embedding = generate_embedding(query)
    similarities = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

def generate_response(query):
    relevant_properties = find_relevant_properties(query)
    context = "Here are some properties that match your query:\n"
    for i, row in relevant_properties.iterrows():
        context += f"- {row['title']} in {row['displayAddress']} with {row['bedrooms']} bedrooms and {row['bathrooms']} bathrooms, priced at {row['price']} AED.\n"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful real estate assistant."},
            {"role": "user", "content": f"{query}\n{context}"}
        ]
    )
    return response.choices[0].message.content

%%writefile app.py
import streamlit as st
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = AzureOpenAI(
    api_key="BqDcDlWbo6D5UYktjh9PAFCHEHAOEh271ALbVTIAzVmz0DD9eNUDJQQJ99BAACHrzpqXJ3w3AAABACOGqLmb",
    api_version="2024-10-21",
    azure_endpoint="https://aldar-test.openai.azure.com/"
)

df = pd.read_csv('/content/uae_real_estate_2024.csv')

df['description'] = df['description'].fillna("")
df = df[df['description'].str.len() > 0]

def generate_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return np.zeros(1536)
        return embedding
    except Exception as e:
        return np.zeros(1536)

df['embedding'] = df['description'].apply(lambda x: generate_embedding(x))

def find_relevant_properties(query, top_k=5):
    query_embedding = generate_embedding(query)
    similarities = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

def generate_response(query):
    relevant_properties = find_relevant_properties(query)
    context = "Here are some properties that match your query:\n"
    for i, row in relevant_properties.iterrows():
        context += f"- {row['title']} in {row['displayAddress']} with {row['bedrooms']} bedrooms and {row['bathrooms']} bathrooms, priced at {row['price']} AED.\n"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful real estate assistant."},
            {"role": "user", "content": f"{query}\n{context}"}
        ]
    )
    return response.choices[0].message.content

st.title("UAE Real Estate RAG Chatbot")
user_query = st.text_input("Enter your query (e.g., 'Find me a 3-bedroom villa in Abu Dhabi'):")
if user_query:
    response = generate_response(user_query)
    st.write(response)

!pkill ngrok

from pyngrok import ngrok

ngrok.set_auth_token("2rfT6qIwjzE9FsItsRHA9L0Mxl4_5t15YrSVLgp8hsSDk7QUY")

!streamlit run app.py &>/dev/null&

public_url = ngrok.connect(addr='8501', proto='http')
print("Your Streamlit app is available at:", public_url)