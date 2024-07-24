# Text Summarizer Flask Application

This Flask application allows users to input text, generate summaries using a BART-based model, and save the inputs and summaries to a SQLite database. It also includes functionality to fine-tune the summarization model with user-provided data.

## Features

- **Text Summarization**: Users can input text, which is then summarized using a fine-tuned BART model.
- **Database Storage**: User inputs and generated summaries are stored in a SQLite database.
- **Model Fine-Tuning**: Optionally fine-tune the summarization model with data from the database.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/text-summarizer.git
    cd text-summarizer
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the model and tokenizer** (if not pre-downloaded):
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    ```

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Open your web browser** and navigate to `http://127.0.0.1:5000/`.

3. **Enter text** in the provided form to get a summary. The summarized text will be displayed and saved in the database.

4. **Model Fine-Tuning**: To fine-tune the model with user data, you can call the `fine_tune_model` function with data retrieved from the database.

## Project Structure

text-summarizer/
- app.py # Main Flask application script
- requirements.txt # List of dependencies
- fine_tuned_model/ # Directory for saving fine-tuned models
- generation_config/ # Directory for saving generation configuration
- README.md # This file


## Dependencies

- Flask
- transformers
- sqlite3
- torch
- re
- os


 ## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
If you have suggestions or improvements, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Contact
For any questions or feedback, please reach out to vishnuvardhanv046@example.com
 
  
  

To install these dependencies, use the `requirements.txt` file:

```plaintext
Flask==2.2.3
transformers==4.27.1
torch==2.0.0


