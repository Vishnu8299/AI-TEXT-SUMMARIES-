import requests
from flask import Flask, render_template, send_file, request
from newspaper import Article
import nltk
from bs4 import BeautifulSoup
import json
from io import BytesIO
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments
import os
import sqlite3
import torch

# Initialize Flask app
app = Flask(__name__)

# Initialize summarization pipeline
MODEL_NAME = 'facebook/bart-large-cnn'
MODEL_PATH = './fine_tuned_model'

summary_generation_config = GenerationConfig(
    max_length=300,
    min_length=100,
    early_stopping=True,
    num_beams=4,
    length_penalty=2.0,
    no_repeat_ngram_size=3,
    forced_bos_token_id=0,
    forced_eos_token_id=2
)

if os.path.exists(MODEL_PATH):
    summarizer = pipeline(
        "summarization",
        model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH),
        generation_config=summary_generation_config
    )
else:
    summarizer = pipeline(
        "summarization",
        model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        generation_config=summary_generation_config
    )

# Download NLTK resources
nltk.download('punkt')

def fetch_article_urls():
    base_url = 'https://www.hindustantimes.com/'
    categories = ['sports']
    all_matching_links = []
    seen_article_ids = set()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for category in categories:
        url = base_url + category
        response = requests.get(url, headers=headers)
        matching_links = []

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            a_tags = soup.find_all('a', href=True)

            for a_tag in a_tags:
                href = a_tag['href']
                if a_tag.has_attr('data-articleid'):
                    article_id = a_tag['data-articleid']
                    if article_id not in seen_article_ids:
                        seen_article_ids.add(article_id)
                        full_url = base_url + href if href.startswith('/') else href
                        matching_links.append({'article_id': article_id, 'url': full_url})
        else:
            print(f'Failed to retrieve the page for {category}. Status code:', response.status_code)

        if matching_links:
            all_matching_links.append({'category': category, 'links': matching_links})

    return all_matching_links

def scrape_articles():
    all_urls = fetch_article_urls()
    articles = {}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for category in all_urls:
        for article in category['links']:
            article_id = article['article_id']
            url = article['url']
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                article_data = Article(url)
                article_data.download()
                article_data.parse()
                article_data.nlp()

                title = article_data.title
                summary = article_data.summary
                publish_date = str(article_data.publish_date) if article_data.publish_date else 'N/A'
                image_url = article_data.top_image

                # Summarize the title and summary
                summarized_title = summarizer(title, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
                summarized_summary = summarizer(summary, max_length=300, min_length=100, do_sample=False)[0]['summary_text']

                articles[article_id] = {
                    'category': category['category'],
                    'url': url,
                    'title': summarized_title,
                    'summary': summarized_summary,
                    'publish_date': publish_date,
                    'image_url': image_url
                }
                
                # Save to database
                with sqlite3.connect('summarizer.db') as conn:
                    conn.execute(
                        'INSERT INTO user_input (article_text, summary_text, title) VALUES (?, ?, ?)', 
                        (summary, summarized_summary, summarized_title)
                    )
            else:
                print(f'Failed to retrieve article from {url}. Status code:', response.status_code)

    return articles

@app.route('/')
def index():
    articles = scrape_articles()
    return render_template('main3x.html', articles=articles)

@app.route('/download')
def download():
    articles = scrape_articles()
    json_data = json.dumps(articles, indent=4)

    json_file = BytesIO()
    json_file.write(json_data.encode('utf-8'))
    json_file.seek(0)

    return send_file(json_file, as_attachment=True, download_name='articles.json', mimetype='application/json')

@app.route('/train', methods=["POST"])
def train():
    # Fetch all data from the database
    data = get_data_from_db()
    if not data:
        return "No data available for training", 400

    # Fine-tune the model with the data
    fine_tune_model(data)
    return "Model trained successfully", 200

def get_data_from_db():
    with sqlite3.connect('summarizer.db') as conn:
        cursor = conn.execute('SELECT article_text, summary_text FROM user_input')
        data = cursor.fetchall()
    return data

def fine_tune_model(data):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_texts, train_summaries = zip(*data)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=1024)
    train_labels = tokenizer(list(train_summaries), truncation=True, padding=True, max_length=150)

    class SummarizationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels['input_ids'][idx])
            return item

        def __len__(self):
            return len(self.labels['input_ids'])

    train_dataset = SummarizationDataset(train_encodings, train_labels)

    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=1,              
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
    )

    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

if __name__ == '__main__':
    app.run(debug=True)
