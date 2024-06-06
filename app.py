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

def llm_analyze(query,user_profiles):
   analysis = client.chat.completions.create(
     model="gpt-4-turbo",
     messages=[
         {"role": "system", "content": """You are a helpful assistant of QNB team. QNB is a Turkish bank.
         You will receive user profile summaries for 10 user at each time along with a search query QNB team wrote.
         These users will be users that are retrieved regarding the query QNB team wrote. 
         You need to post filter the search results are response the query by using user profile summaries you receive."""},
         {"role": "user", "content": f""" Query: {query}/n/n User Profiles:{user_profiles}"""} ])
   print(analysis)
   return analysis


# Function to query the Pinecone API
def query_pinecone(user_profiles):
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
            final_result = llm_analyze(user_input,summary_list)
            # Display the final answer
            st.write("LLM Analysis:", summary_list)
            st.write("Search Results:", summary_list)
        except KeyError as e:
            st.error(f"KeyError: {e}. The expected key was not found in the response.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a query.")
