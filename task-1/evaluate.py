import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langgraph.graph import END

# Import our compiled agent 'app' from the app.py file
from app import app, AgentState 

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Import the RAGAs wrappers for LangChain models
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Load all our API keys from the .env file
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

print("Initializing evaluation models...")

# --- 1. Initialize Models for RAGAs ---
# RAGAs needs its own LLM and Embeddings to perform the evaluation.
# We'll use the same stack as our agent (Gemini and HF) to be consistent.

# We wrap our Gemini model so RAGAs can use it as a "judge"
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)
ragas_llm = LangchainLLMWrapper(llm)

# We do the same for our embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

print(">>> Starting RAGAS evaluation...")

# --- 2. Define our Test Set ---
# This is our "ground truth" data to test the agent against.
test_questions = [
    "What are the two pre-training tasks for BERT?",
    "What is the Transformer architecture based on, and what does it not use?",
    "What two networks does AlphaGo use?",
]
ground_truths = [
    "BERT is pre-trained on two tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP).",
    "The Transformer is based on self-attention mechanisms and does not use RNNs or convolution.",
    "AlphaGo uses a policy network to select moves and a value network to estimate the likelihood of winning.",
]

# --- 3. Run the Agent ---
print("\n>>> Executing agent to generate answers...")
results = []
for i, q in enumerate(test_questions):
    print(f"\nQuestion {i+1}/{len(test_questions)}: {q}")
    
    # This is the same input we'd give the agent in the UI
    state_inputs = {"question": q, "loop_count": 0}
    
    # We use .invoke() to run the agent from start to finish
    # and get the complete final state dictionary.
    final_state = app.invoke(state_inputs, config={"recursion_limit": 5})

    # Get the answer and the documents from the final state
    generated_answer = final_state.get("answer", "No answer generated.")
    retrieved_docs = final_state.get("documents", [])

    print(f"Generated answer: {generated_answer[:100]}...")

    # Store everything we need for RAGAs
    results.append({
        "question": q,
        "answer": generated_answer,
        "ground_truth": ground_truths[i],
        "contexts": retrieved_docs  # We'll rename this key in the next step
    })

print("\n>>> Evaluating answers using RAGAS metrics...")

# --- 4. Prepare the Dataset for RAGAs ---
# RAGAs is particular about column names, so we need to create
# a dictionary that matches what it expects.
dataset_dict = {
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "retrieved_contexts": [r["contexts"] for r in results], # RAGAs needs this exact name
    "ground_truth": [r["ground_truth"] for r in results],
}
# Convert our list of results into a Hugging Face Dataset object
eval_data = Dataset.from_dict(dataset_dict)

# --- 5. Run the Evaluation ---
score = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

# --- 6. Display Results ---
print("\n>>> Evaluation completed. Summary of RAGAS scores:\n")
# Convert the RAGAs output to a pandas DataFrame for easy reading
df_results = score.to_pandas()
print(df_results)

# Save the scores to a file for our records
df_results.to_csv("ragas_evaluation_results.csv", index=False)
print("\nEvaluation results saved to 'ragas_evaluation_results.csv'")