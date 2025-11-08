# LangGraph Self-Correcting RAG Agent
### As part of SwarmLens Generative AI Engineer Internship – Test Submission

**Created by: Arnav Chauhan** **3rd Year UG** **Indian Institute of Technology, Kharagpur**

---

This project implements an advanced Q&A AI agent using LangGraph. The agent answers technical questions from a knowledge base of seminal AI research papers and includes a full suite of bonus features: a Streamlit UI, LangSmith tracing, and RAGAs evaluation.

## ⚙️ Setup

1.  **Clone the Repository:**
    First, clone this repository to your local machine. This repository includes the `knowledge_base` directory with the required PDF files.
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project requires all libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys (in .env file):**
    Create a file named `.env` in the root of the directory. You will need a **Google API Key** (for the Gemini LLM) and a **Hugging Face Token** (for the embeddings model). For the bonus features, you also need a LangSmith API key.
    ```
    # Main LLM Key (Gemini)
    GOOGLE_API_KEY="..."

    # Embeddings Key (Hugging Face)
    HF_TOKEN="..."

    # Bonus 2: LangSmith Tracing
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_API_KEY="ls__..."
    LANGCHAIN_PROJECT="Aatoon-task" 
    ```

---

## 🚀 How to Run

You can run the project in three different modes:

### 1. Run the Interactive UI (Bonus 1)
This is the main way to interact with the agent.
```bash
streamlit run streamlit_app.py

This will open a chat window in your browser. You can ask questions and see the agent's internal thoughts (plan, retrieve, reflect) in real-time.

2. Run the Evaluation (Bonus 4)

This will test the agent against a "ground truth" dataset and score it using RAGAs.
Bash

python evaluate.py

This will print a table of scores (faithfulness, relevancy, etc.) and save the results to ragas_evaluation_results.csv.

3. Run the Console App

This runs the original app.py script directly and prints all logs to the console.
Bash

python app.py