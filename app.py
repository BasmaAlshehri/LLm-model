from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
import requests
import logging
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from dotenv import load_dotenv
import os
from flask_cors import CORS


load_dotenv()

app = Flask(__name__, template_folder=r'D:\New folder\PyCharm 2023.3.3\pythonProject1\templates')
app.debug = True
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
FDA_API_URL = 'https://api.fda.gov/drug/label.json'
logging.basicConfig(level=logging.DEBUG)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})  # Adding a pad token if it's not set
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to match tokenizer configuration

model_path = r'D:\New folder\PyCharm 2023.3.3\pythonProject1\templates\Lastmodel2.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load the model: {e}")

# RSA key pair generation and encryption/decryption functions
def generate_rsa_key_pair():
    key = RSA.generate(2048)
    return key.export_key(), key.publickey().export_key()

def encrypt_data(data, public_key):
    key = RSA.import_key(public_key)
    cipher = PKCS1_OAEP.new(key)
    return cipher.encrypt(data.encode())

def decrypt_data(encrypted_data, private_key):
    key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(key)
    return cipher.decrypt(encrypted_data).decode()

# User authentication setup
users = {'doctor': generate_password_hash('secure123')}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/scenario1', methods=['GET', 'POST'])
def scenario1():
    if request.method == 'POST':
        try:
            data = request.get_json()
            drug_names = data.get('drugNames', [])
            if not drug_names:
                app.logger.debug("No drug names provided.")
                return jsonify({'error': 'Invalid drugNames, must be a list of names'}), 400

            inputs = tokenizer(drug_names, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).item()

            if predictions == 1:
                result = "Drug-Drug interaction detected."
            else:
                response = requests.get(f"{FDA_API_URL}?search={'+AND+'.join(drug_names)}&limit=1")
                if response.status_code == 200:
                    result = "FDA data checked: Possible interaction found."
                else:
                    result = "No Drug-Drug interaction detected and no FDA data available."

            return jsonify({'prediction': result})
        except Exception as e:
            app.logger.exception("An error occurred during prediction.")
            return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500
    else:
        return render_template('scenario1.html')


@app.route('/scenario2', methods=['GET', 'POST'])
def scenario2():
    if 'authenticated' not in session:
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            user_hash = users.get(username)
            if user_hash and check_password_hash(user_hash, password):
                session['authenticated'] = True
                return redirect(url_for('scenario2'))
            else:
                return render_template('scenario2.html', error="Invalid username or password")
        return render_template('scenario2.html')
    else:
        if request.method == 'POST':
            patient_name = request.form.get('patientName')
            drug_name = request.form.get('drugName')
            drug_names = [patient_name, drug_name]
            inputs = tokenizer(drug_names, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            result = "Drug-Drug interaction detected." if predictions.item() == 1 else "No Drug-Drug interaction."
            return jsonify({'prediction': result})
        return render_template('scenario2.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
