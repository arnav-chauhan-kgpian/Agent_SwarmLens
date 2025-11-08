import os
from dotenv import load_dotenv
from typing import List, Literal
from typing_extensions import TypedDict

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# Using the new langchain-huggingface package for embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------
# 1. Environment and Model Initialization
# ---------------------------------------------------------

# Load API keys and other secrets from a .env file
load_dotenv()

# Set environment variables for the services we're using
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY') 
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN') 

print("Initializing language model and embeddings...")

# Initialize our LLM, using Gemini 1.5 Flash for its speed and capability
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)

# Initialize our embeddings model, using a popular and efficient one from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 2. Retrieval-Augmented Generation (RAG) Setup
# ---------------------------------------------------------

def setup_rag_retriever(dir_path: str):
    """
    Load PDFs from a directory, split them into chunks, embed them, 
    and build a retriever on top of a Chroma vector store.
    """
    print(f"\nLoading PDF documents from: {dir_path}")
    # Load all PDF files from the specified directory
    loader = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDF documents found in '{dir_path}'.")

    # Split the loaded documents into smaller, manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} text chunks.")

    # Create a Chroma vector store from the chunks and persist it to disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_papers" # This saves the DB locally
    )

    # Create a retriever that will fetch the top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("Retriever setup complete.")
    return retriever


# Set up the retriever as soon as the app starts
retriever = setup_rag_retriever("knowledge_base")


# ---------------------------------------------------------
# 3. Agent State Definition
# ---------------------------------------------------------

class AgentState(TypedDict):
    """
    This is our "shared memory" for the agent. 
    It's a dictionary that gets passed between all the nodes.
    """
    question: str
    rewritten_query: str
    documents: List[str]
    answer: str
    reflection_critique: str
    reflection_decision: str
    loop_count: int


# ---------------------------------------------------------
# 4. Structured Output Schemas
# ---------------------------------------------------------

# Using Pydantic models helps us get structured JSON output from the LLM
class QueryRewriter(BaseModel):
    """A schema for the rewritten search query."""
    rewritten_query: str = Field(description="Optimized version of the user query.")


class Reflection(BaseModel):
    """A schema for the self-reflection step."""
    critique: str = Field(description="Feedback on the generated answer.")
    decision: Literal["Good", "Revise"] = Field(description="Evaluation outcome.")


# ---------------------------------------------------------
# 5. Node Implementations
# ---------------------------------------------------------

def plan_node(state: AgentState):
    """
    First node: Rewrites the user's question into an optimized query.
    If we're in a "revise" loop, it uses the critique to improve the query.
    """
    print("\nExecuting PLAN node...")
    
    # Use .with_structured_output to force Gemini to return a clean QueryRewriter object
    structured_llm = llm.with_structured_output(QueryRewriter)

    # If we've looped back, use the critique to make a better query
    if state.get("reflection_critique"):
        prompt_template = PromptTemplate(
            template="""You are a query rewriting expert.
A user asked: {question}
Critique of previous attempt: {critique}
Based on this feedback, rewrite the question into a more effective search query.""",
            input_variables=["question", "critique"],
        )
        prompt = prompt_template.format(
            question=state["question"],
            critique=state["reflection_critique"]
        )
    else:
        # This is the first pass
        prompt_template = PromptTemplate(
            template="""You are a query rewriting expert.
Rewrite the following question into a concise, optimized search query.
Question: {question}""",
            input_variables=["question"],
        )
        prompt = prompt_template.format(question=state["question"])

    # Call the LLM with the prompt
    result = structured_llm.invoke(prompt)

    print(f"Rewritten query: {result.rewritten_query}")
    # Update the state
    return {
        "rewritten_query": result.rewritten_query,
        "loop_count": state.get("loop_count", 0) + 1
    }


def retrieve_node(state: AgentState):
    """
    Second node: Uses the rewritten query to fetch documents from ChromaDB.
    """
    print("\nExecuting RETRIEVE node...")
    query = state["rewritten_query"]
    
    # Get the relevant document chunks
    results = retriever.invoke(query)
    # Just save the text content to the state
    documents = [d.page_content for d in results]
    
    print(f"Retrieved {len(documents)} document chunks.")
    return {"documents": documents}


def answer_node(state: AgentState):
    """
    Third node: Generates an answer using the retrieved documents as context.
    """
    print("\nExecuting ANSWER node...")
    prompt_template = PromptTemplate(
        template="""You are an AI research assistant.
Answer the user's question *only* based on the provided context.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"],
    )

    # Combine all the document chunks into a single string
    context_text = "\n\n---\n\n".join(state["documents"])
    
    # We just want a string answer, so no structured output needed here
    chain = prompt_template | llm
    
    # The .invoke() call returns a message object, so we grab its .content
    answer_message = chain.invoke({"context": context_text, "question": state["question"]})

    print(f"Generated answer preview: {answer_message.content[:100]}...")
    return {"answer": answer_message.content}


def reflect_node(state: AgentState):
    """
    Fourth node: The "LLM-as-Judge" step.
    Critiques the generated answer and decides whether to loop.
    """
    print("\nExecuting REFLECT node...")
    
    # We'll use structured output again to get a reliable Reflection object
    structured_llm = llm.with_structured_output(Reflection)

    prompt_template = PromptTemplate(
        template="""You are an evaluator reviewing an AI-generated answer.
Check for relevance and factual correctness using the provided context.

User Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Provide a brief critique and a decision ("Good" or "Revise").""",
        input_variables=["question", "context", "answer"],
    )

    # We need to chain the prompt template to the structured LLM
    chain = prompt_template | structured_llm

    context_text = "\n\n---\n\n".join(state["documents"])
    
    # Now, invoke the chain with the inputs
    reflection = chain.invoke({
        "question": state["question"],
        "context": context_text,
        "answer": state["answer"]
    })

    print(f"Critique: {reflection.critique}")
    print(f"Decision: {reflection.decision}")
    
    # Update the state with the reflection
    return {
        "reflection_critique": reflection.critique,
        "reflection_decision": reflection.decision
    }


# ---------------------------------------------------------
# 6. Looping Logic
# ---------------------------------------------------------

def should_loop(state: AgentState):
    """
    This is the conditional edge that decides where to go after reflection.
    """
    print("\nEvaluating loop condition...")
    
    # Safety break to prevent infinite loops
    if state["loop_count"] >= 2:
        print("Loop limit reached. Terminating process.")
        return END
        
    # If the decision was "Revise", go back to the 'plan' node
    if state["reflection_decision"] == "Revise":
        print("Reflection decision: Revise → looping back to PLAN node.")
        return "plan"
        
    # Otherwise, the answer is "Good", so we end the graph
    print("Reflection decision: Good → workflow complete.")
    return END


# ---------------------------------------------------------
# 7. LangGraph Workflow Construction
# ---------------------------------------------------------

print("\nBuilding LangGraph workflow...")
workflow = StateGraph(AgentState)

# Add all our functions as nodes in the graph
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer", answer_node)
workflow.add_node("reflect", reflect_node)

# Define the connections between the nodes
workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "reflect")

# This is the special conditional edge that enables our loop
workflow.add_conditional_edges(
    "reflect",  # The starting node
    should_loop, # The function that makes the decision
    {
        "plan": "plan", # If it returns "plan", go to the 'plan' node
        END: END        # If it returns END, finish the graph
    }
)

# Compile the graph into a runnable application
app = workflow.compile()
print("Graph compiled successfully.")


# ---------------------------------------------------------
# 8. Main Execution Block
# ---------------------------------------------------------

# This block only runs when you execute `python app.py` directly
if __name__ == "__main__":
    """
    Entry point for running the reflective RAG agent from console.
    """
    print("\nRunning the reflective RAG agent...")

    sample_question = "What are the two pre-training tasks for BERT?"
    
    # Use app.invoke() to run the full graph and get the final state.
    # This is perfect for a console script.
    final_output = app.invoke(
        {"question": sample_question, "loop_count": 0}, 
        config={"recursion_limit": 5}
    )

    print("\nFINAL ANSWER:")
    # The final_output is the complete agent state, so we can .get() the answer.
    print(final_output.get("answer", "No answer generated."))