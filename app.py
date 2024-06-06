import streamlit as st
import requests
import os
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("qnbeyond")

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

# Function to query the ChatGPT API
# Define the function to create an embedding
def create_embedding(user_input):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input,
        encoding_format="float"
    )
    #embd = response.object.data[0].embedding
    return response.data[0].embedding

# Function to query the Pinecone API
def query_pinecone(vector):
    q_result = index.query(
        vector=vector,
        include_values=False,
        include_metadata=True,
        top_k=10,
    )

    return q_result

# Streamlit app
st.title("QNB Vector Search Demo App")

# User input
user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        # Query ChatGPT
        embedding = create_embedding(user_input)
        try:
            # Query Pinecone
            pinecone_response = query_pinecone(embedding)
            summary_list = [i["metadata"]["summary"] for i in pinecone_response['matches']]

            # Display the final answer
            st.write("Pinecone Response:", summary_list)
        except KeyError as e:
            st.error(f"KeyError: {e}. The expected key was not found in the response.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a query.")