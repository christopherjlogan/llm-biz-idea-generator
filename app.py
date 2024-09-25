import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Use OpenAI API Key')
parser.add_argument('--api-key', required=True, help='OpenAI API key')
parser.add_argument('--openai-model', required=True, help='OpenAI API key')
args = parser.parse_args()
api_key = args.api_key
openai.api_key = args.api_key
# Set OpenAI context
client = OpenAI(
  api_key=openai.api_key,
)
openai_model=args.openai_model

# Function to calculate cosine similarity between two embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    # Reshape embeddings to 2D arrays
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# Function to get embeddings for a list of texts
def get_embeddings(client, texts, model='text-embedding-3-small'):
  #texts = texts.replace("\n", " ")
  return client.embeddings.create(input = [texts], model=model).data[0].embedding

# Function to interact with ChatGPT using the new API
def chat_with_assistant(client, messages, model=openai_model):
    # Format messages for the new API
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    assistant_reply = response.choices[0].message.content
    return assistant_reply

# Evaluate training text
print("Evaluating training data")
with open('training.txt', 'r', encoding='utf-8') as f:
  training_text = f.read()
training_embedding = get_embeddings(client, training_text)

# Get business idea
print("Reading in business idea")
with open('business_idea.txt', 'r', encoding='utf-8') as f:
  initial_business_idea = f.read()

# Evaluating prompt catalogs
print("Evaluating prompt catalogs")
directory="catalog"
scores = []
for filepath in os.listdir(directory):
  filename = os.path.join(directory, filepath)
  if os.path.isfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    conversation = []
    all_prompts = []
    assistant_responses = []
    combined_texts = []

    # Start the conversation with the initial business idea
    business_idea_prefix="I'm going to ask you a series of questions to guide development of a startup pitch for the business idea.  I only need your acknowledgement at this point: "
    conversation.append({"role": "user", "content": business_idea_prefix + initial_business_idea})

    # Get assistant's initial response
    assistant_reply = chat_with_assistant(client, conversation)
    conversation.append({"role": "assistant", "content": assistant_reply})

    # Print the initial business idea and assistant's response
    print(f"Initial Business Idea: {initial_business_idea}")
    print(f"Assistant Response:\n{assistant_reply}\n")

    # Store the initial interaction
    all_prompts.append(initial_business_idea)
    assistant_responses.append(assistant_reply)
    combined_text = f"User: {initial_business_idea}\nAssistant: {assistant_reply}"
    combined_texts.append(combined_text)

    # Process each prompt from 'promptcatalog.txt'
    print("Processing prompts and assistant responses...")
    assistant_reply=""
    for i, prompt in enumerate(prompts):
        # Append user's prompt to the conversation
        conversation.append({"role": "user", "content": prompt})

        # Get assistant's response
        assistant_reply = chat_with_assistant(client, conversation)

        # Append assistant's response to the conversation
        conversation.append({"role": "assistant", "content": assistant_reply})

        # Print the prompt and the assistant's response
        print(f"Prompt {i+1}: {prompt}")
        print(f"Assistant Response:\n{assistant_reply}\n")

        # Store the prompt and assistant response
        all_prompts.append(prompt)
        assistant_responses.append(assistant_reply)

        # Combine prompt and response for embedding
        combined_text = f"User: {prompt}\nAssistant: {assistant_reply}"
        combined_texts.append(combined_text)
        #prompt_embeddings = get_embeddings(client, assistant_reply)

    print("Generating embeddings...")
    embeddings = get_embeddings(client, assistant_reply)
    similarity = calculate_cosine_similarity(training_embedding, embeddings)
    print(filename + " similarity to training data: ", similarity)
    scores.append(filename + ": " + str(similarity))

print("Catalog evaluation summary:")
for score in scores:
  print(score)
