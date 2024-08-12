from flask import Flask, request, render_template_string
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments
import sqlite3
import os
import torch

app = Flask(__name__)

DATABASE = 'summarizer.db'
MODEL_NAME = 'facebook/bart-large-cnn'
MODEL_PATH = './fine_tuned_model'

# Define generation configurations
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

title_generation_config = GenerationConfig(
    max_length=30,
    min_length=5,
    early_stopping=True,
    num_beams=2,
    length_penalty=1.0,                                                                                                                                 
    no_repeat_ngram_size=2,
    forced_bos_token_id=0,
    forced_eos_token_id=2
)

# Save the GenerationConfig to files
summary_generation_config.save_pretrained('./generation_config_summary')
title_generation_config.save_pretrained('./generation_config_title')

# Initialize summarization pipeline
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

# Initialize title generation pipeline
title_generator = pipeline(
    "summarization",
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    generation_config=title_generation_config
)

# Initialize database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        # Drop the existing table if it exists (for development/testing purposes)
        conn.execute('DROP TABLE IF EXISTS user_input')
        # Create the table with the correct schema
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_input (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_text TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                title TEXT NOT NULL
            )
        ''')

def update_db_schema():
    with sqlite3.connect(DATABASE) as conn:
        # Check if the 'title' column exists
        cursor = conn.execute('PRAGMA table_info(user_input)')
        columns = [row[1] for row in cursor.fetchall()]
        if 'title' not in columns:
            conn.execute('''
                ALTER TABLE user_input
                ADD COLUMN title TEXT NOT NULL DEFAULT '';
            ''')

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    title = ""
    if request.method == "POST":
        article_text = request.form.get("article_text", "")
        
        if article_text:
            # Generate summary
            result = summarizer(article_text, max_length=300, min_length=100, do_sample=False)
            summary = result[0]['summary_text']
            
            # Generate title from the summary
            title_result = title_generator(summary, max_length=45, min_length=18, do_sample=False)
            title = title_result[0]['summary_text'].strip()
            
            # Ensure the title does not exceed 12 words
            title_words = title.split()
            if len(title_words) > 12:
                title = ' '.join(title_words[:12]) + '.'

            # Save user input, title, and summary to database
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    'INSERT INTO user_input (article_text, summary_text, title) VALUES (?, ?, ?)', 
                    (article_text, summary, title)
                )
    return render_template_string("""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Text Summarizer</title>
      </head>
      <body>
        <h1>Text Summarizer</h1>
        <form method="post">
          <textarea name="article_text" rows="10" cols="80" placeholder="Enter text to summarize...">{{ request.form.get('article_text', '') }}</textarea><br>
          <input type="submit" value="Summarize">
        </form>
        <h2>Title:</h2>
        <textarea rows="1" cols="80" readonly>{{ title }}</textarea>
        <h2>Summary:</h2>
        <textarea rows="10" cols="80" readonly>{{ summary }}</textarea>
      </body>
    </html>
    """, summary=summary, title=title)

@app.route("/train", methods=["POST"])
def train():
    # Fetch all data from the database
    data = get_data_from_db()
    if not data:
        return "No data available for training", 400

    # Fine-tune the model with the data
    fine_tune_model(data)
    return "Model trained successfully", 200

def get_data_from_db():
    with sqlite3.connect(DATABASE) as conn:
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

if __name__ == "__main__":
    init_db()  # Initialize the database and table schema
    update_db_schema()  # Ensure the schema is up-to-date
    app.run(debug=True)
