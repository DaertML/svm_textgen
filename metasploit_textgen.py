# Import libraries
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.svm import SVR
import sys


# Define a function to get gpt2 embeddings for a sentence
def get_gpt2_embeddings(sentence):
  # Initialize the tokenizer and model
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2Model.from_pretrained('gpt2')
  # Tokenize the sentence and get the input ids
  input_ids = tokenizer.encode(sentence, return_tensors='pt')
  # Get the hidden states of the last layer
  outputs = model(input_ids)
  last_hidden_states = outputs[0]
  # Average the hidden states along the sequence dimension
  embeddings = torch.mean(last_hidden_states, dim=1)
  # Return a numpy array
  return embeddings.detach().numpy()

# Create some sentences for the training data
sentences = open("./msf.txt", "r").readlines()

# Get the gpt2 embeddings for each sentence
X = np.concatenate([get_gpt2_embeddings(sentence) for sentence in 
sentences])

# Define the target values as the next sentence index
y = np.arange(1, len(sentences) + 1) % len(sentences)

# Train a SVM model to predict the next sentence index
model = SVR()
model.fit(X, y)

# Define a function to generate a sentence given a prompt sentence
def generate_sentence(prompt):
  # Get the gpt2 embeddings for the prompt sentence
  X = get_gpt2_embeddings(prompt)
  # Predict the next sentence index
  y = model.predict(X)
  # Round the index to the nearest integer
  index = int(round(y[0]))
  # Return the corresponding sentence
  return sentences[index]

# Test the model with a prompt sentence
prompt = "How to run a exploit?"
print("Prompt:", prompt)
print("Generated:", generate_sentence(prompt))
