# Insider Threat Tweet Analyzer Web App

The MVP utilizes a combination of well-known hugging face piplines for NLP analysis and Qwen 2.5 (3B parameters) for LLM analysis. 

Feel free to test the MVP with the provided .xlsx file. Do not use the large .xlsx files given by the POC, the provided file (in the test_xlsx directory) contains only ten tweets vs the 50,000+ in the given datasets. You may add to the template of tweets (just a copy of POC provided format with less tweets). The final analysis will be returned as a summary through data visualization, but also an excel is made available for download.

## Frontend Setup Instructions

Follow these steps to set up and run the frontend:

1. **Navigate to the project directory:**

    ```bash
    cd Insider-Threat-Analyzer-MVP
    ```

2. **Install dependencies:**

    ```bash
    npm install
    ```

3. **Start the frontend development server:**

    ```bash
    npm run dev
    ```


## Backend Setup Instructions

Follow these steps to set up and run the backend:

1. **Navigate to the server directory:**

    ```bash
    cd server
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

5. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Start the backend server:**

    ```bash
    python3 app.py
    ```

## Huberman Podcast RAG

This project contains a simple Retrieval-Augmented Generation pipeline for answering questions about Huberman Lab podcast transcripts. It runs completely on-device once models are downloaded.

### Quick Start

```bash
make setup       # install python deps
make index       # build FAISS index from transcripts
python main.py --question "How do I return to sleep?" --model <path/to/model>
```

To start the web interface:

```bash
make run
```

Then open <http://localhost:11434> in your browser.
