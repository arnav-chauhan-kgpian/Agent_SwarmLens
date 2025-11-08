import streamlit as st
from langgraph.graph import END

# Import our compiled agent 'app' and the 'AgentState' from our main app.py file
from app import app, AgentState

# --- 1. Configure the Streamlit Page ---
st.set_page_config(page_title="🤖 AI Research Agent", layout="wide")
st.title("🤖 AI Research Agent")
st.caption("Query an AI research assistant trained on papers such as Transformers, BERT, and AlphaGo.")

# --- 2. Set up Chat History ---
# We use session_state, which is Streamlit's way of remembering things
# between button clicks and reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize an empty list for our chat

# --- 3. Display Past Messages ---
# Loop through all the messages we've saved and display them on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. Define the Agent Streaming Function ---
def stream_agent_response(user_question):
    """
    This function takes the user's question and streams the agent's
    thoughts and final answer back to the UI, one piece at a time.
    """
    # This is the starting "memory" for our agent's run
    init_state = {"question": user_question, "loop_count": 0}
    
    # We use .stream() to get live updates from the LangGraph agent
    for evt in app.stream(init_state, config={"recursion_limit": 5}):
        # Get the name of the node that just ran
        node_name = list(evt.keys())[0]
        
        # --- Show the user what the agent is thinking ---
        if node_name == "plan":
            rewritten = evt[node_name].get("rewritten_query", "")
            yield f"### 💡 Planning\nRewriting query for better understanding: `{rewritten}`\n"
            
        elif node_name == "retrieve":
            docs = evt[node_name].get("documents", [])
            doc_count = len(docs)
            yield f"### 📚 Document Retrieval\nLocated {doc_count} relevant text segments.\n"
            
        elif node_name == "answer":
            final_answer = evt[node_name].get("answer", "")
            # We yield the full answer here
            yield f"### 💬 Generating Answer\nConstructing final response based on retrieved documents.\n\n---\n{final_answer}\n\n---\n"
            
        elif node_name == "reflect":
            critique_txt = evt[node_name].get("reflection_critique", "N/A")
            reflection_choice = evt[node_name].get("reflection_decision", "Unknown")
            yield f"### 🤔 Reflection Phase\n**Critique:** {critique_txt}\n**Decision:** `{reflection_choice}`\n"
            
            # Let the user know if the agent is trying again
            if reflection_choice.lower() == "revise":
                yield "\n---\nRevision triggered — refining the previous answer.\n---\n"
                
        elif node_name == END:
            yield "\n✅ Process complete. Agent workflow finished successfully."

# --- 5. Handle User Input ---
# This is the main interaction loop. It waits for the user to type something.
if user_prompt := st.chat_input("Ask a question (e.g., 'Explain multi-head attention')"):
    
    # First, save and display the user's own message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Now, get and display the agent's response
    with st.chat_message("assistant"):
        # st.write_stream is the magic that displays the output as it comes in
        complete_reply = st.write_stream(stream_agent_response(user_prompt))
        
    # Finally, save the agent's full response to our history
    st.session_state.messages.append({"role": "assistant", "content": complete_reply})