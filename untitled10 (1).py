!pip install pycryptodome
!pip -q install langchain
!pip -q install bitsandbytes accelerate xformers einops
!pip -q install datasets loralib sentencepiece
!pip -q install pypdf
!pip install torch
!pip -q install sentence_transformers
!pip install accelerate>=0.21.0
!pip install transformers
!pip install pandas
!pip install torch
!pip install scikit-learn
!pip install pycryptodome
!pip install gdown
!pip install nltk
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
import requests
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import requests
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import requests
import nltk
import random
from nltk.util import ngrams
from collections import defaultdict, Counter
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

#Training the model : 

import pandas as pd
import gdown
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import os
from google.colab import drive

# Function to load dataset from file
def load_dataset_from_file(file_path):
    df = pd.read_csv(file_path)
    return df

# Google Drive file IDs
scenario1_file_id = ""**""
scenario2_file_id = "**"

# Download files from Google Drive using file IDs
scenario1_file_url = f"https://drive.google.com/uc?id={scenario1_file_id}"
scenario2_file_url = f"https://drive.google.com/uc?id={scenario2_file_id}"

scenario1_file_path = "/content/scenario1.csv"
scenario2_file_path = "/content/scenario2.csv"

gdown.download(scenario1_file_url, scenario1_file_path, quiet=False)
gdown.download(scenario2_file_url, scenario2_file_path, quiet=False)

# Load datasets from local files
scenario1_df = load_dataset_from_file(scenario1_file_path)
scenario2_df = load_dataset_from_file(scenario2_file_path)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer with the specified pad token ID
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token=str(50257))

# Tokenize the input text for scenario 1
train_encodings_scenario1 = tokenizer(scenario1_df['Name1'].tolist(), scenario1_df['Name2'].tolist(), truncation=True, padding=True)
train_labels_scenario1 = torch.tensor(scenario1_df['DLabel'].tolist())
# Create PyTorch dataset for scenario 1
train_dataset_scenario1 = TensorDataset(torch.tensor(train_encodings_scenario1['input_ids']),
                                        torch.tensor(train_encodings_scenario1['attention_mask']),
                                        train_labels_scenario1)
# Define data loader for scenario 1
batch_size = 16  # adjust batch size as needed
train_dataloader_scenario1 = DataLoader(train_dataset_scenario1, batch_size=batch_size, shuffle=True)

# Tokenize the input text for scenario 2

input_text_scenario2 = scenario2_df['PName'] + " " + scenario2_df['Name']
train_encodings_scenario2 = tokenizer(input_text_scenario2.tolist(), truncation=True, padding=True)
train_labels_scenario2 = torch.tensor(scenario2_df['DLabel'].tolist())
# Create PyTorch dataset for scenario 2
train_dataset_scenario2 = TensorDataset(torch.tensor(train_encodings_scenario2['input_ids']),
                                        torch.tensor(train_encodings_scenario2['attention_mask']),
                                        train_labels_scenario2)
# Define data loader for scenario 2
train_dataloader_scenario2 = DataLoader(train_dataset_scenario2, batch_size=batch_size, shuffle=True)

# Load the pre-trained GPT2 model for sequence classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to match the tokenizer's vocabulary size
model.to(device)  # Move the model to the appropriate device

# Ensure that the model's configuration has the pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader_scenario1))

# Training loop for scenario 1
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader_scenario1:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional: gradient clipping
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader_scenario1)
    print(f'Scenario 1 - Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

# Training loop for scenario 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader_scenario2:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional: gradient clipping
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader_scenario2)
    print(f'Scenario 2 - Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

# Specify the directory path where the notebook file is located
notebook_directory = '/content/drive/My Drive/Colab Notebooks'

# Save the model to the notebook directory
save_path = os.path.join(notebook_directory, "Lastmodel.pth")
torch.save(model.state_dict(), save_path)
print("Model saved successfully!" if os.path.exists(save_path) else "Failed to save the model.")


##Evaluation : 

import pandas as pd
from transformers import GPT2ForSequenceClassification, GPT2Config, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import io

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, pretrained_tokenizer):
    # Load the pretrained GPT2 model for sequence classification
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)

    # Load the trained model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    # Investigate the error message
    print("Investigating the error message:")
    print("Model Configuration:")
    print(model.config)

    # Load the checkpoint configuration
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Checkpoint Configuration:")
    print(checkpoint['config'].vocab_size)

    # Compare vocabulary sizes
    if checkpoint['config'].vocab_size != model.config.vocab_size:
        print("Resizing the embedding layer...")
        # Resize the embedding layer
        model.resize_token_embeddings(checkpoint['config'].vocab_size)
        # Load the checkpoint again
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        print("Embedding layer resized and checkpoint loaded.")

    model.eval()
    return model

def load_dataset_from_url(url):
    # Load dataset from URL
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df

def evaluate_model(model, test_data, scenario, tokenizer, batch_size=16):
    if scenario == 1:
        # Tokenize input text for scenario 1
        test_encodings = tokenizer(test_data['Name1'].tolist(), test_data['Name2'].tolist(), truncation=True, padding=True)
    elif scenario == 2:
        # Tokenize input text for scenario 2
        test_encodings = tokenizer(test_data['PName'].tolist(), test_data['DName'].tolist(), truncation=True, padding=True)

    # Create PyTorch dataset for test data
    test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                                  torch.tensor(test_encodings['attention_mask']),
                                  torch.tensor(test_data['DLabel'].tolist()))

    # Define data loader for test data
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over test data batches
    for batch in test_dataloader:
        input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predicted labels
        predicted_class = torch.argmax(logits, dim=-1)

        # Append true and predicted labels
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted_class.tolist())

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=['No Interaction', 'Interaction'])
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

def main():
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Specify the path to your trained model
    model_path = "/content/drive/My Drive/Colab Notebooks/Lastmodel.pth"

    # Load datasets for evaluation
    scenario1_test_url = "*****"
    scenario2_test_url = "*****"

    scenario1_test_data = load_dataset_from_url(scenario1_test_url)
    scenario2_test_data = load_dataset_from_url(scenario2_test_url)

    # Load your trained model
    model = load_model(model_path, tokenizer)

    # Evaluate the model for scenario 1
    print("Evaluation for Scenario 1:")
    evaluate_model(model, scenario1_test_data, scenario=1, tokenizer=tokenizer)

    # Evaluate the model for scenario 2
    print("\nEvaluation for Scenario 2:")
    evaluate_model(model, scenario2_test_data, scenario=2, tokenizer=tokenizer)

if __name__ == "__main__":
    main()


