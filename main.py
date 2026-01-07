import os
import re
import json
import requests
from typing import List, Dict, Any, TypedDict, Literal

# --- LangChain & RAG ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun

# --- LangGraph ---
from langgraph.graph import StateGraph, END

# --- UI ---
import gradio as gr
from dotenv import load_dotenv

# Load Env
load_dotenv()

# ==========================================
# TASK 2: RAG SYSTEM
# ==========================================

class RAGSystem:
    def __init__(self, pdf_path: str, store_dir: str = "./faiss_store"):
        self.pdf_path = pdf_path
        self.store_dir = store_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = self._initialize_store()
        
    def _initialize_store(self):
        if os.path.exists(os.path.join(self.store_dir, "index.faiss")):
            print("✅ Loading existing FAISS index...")
            return FAISS.load_local(self.store_dir, self.embeddings, allow_dangerous_deserialization=True)
        
        print(f"⚠️ Creating dummy PDF at {self.pdf_path}...")
        self._create_dummy_pdf()
        
        print("📄 Ingesting PDF into Vector Store...")
        from pypdf import PdfReader
        reader = PdfReader(self.pdf_path)
        docs = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text:
                docs.append(Document(page_content=text, metadata={"source": os.path.basename(self.pdf_path), "page": i}))
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        vs = FAISS.from_documents(chunks, self.embeddings)
        vs.save_local(self.store_dir)
        return vs

    def _create_dummy_pdf(self):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        text = """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.
        Machine Learning (ML) is a subset of AI focused on building systems that learn from data.
        Neural Networks are computing systems vaguely inspired by the biological neural networks.
        Deep Learning uses multiple layers to progressively extract higher-level features from raw input.
        Natural Language Processing (NLP) enables computers to understand and interpret human language.
        """
        for line in text.split('. '):
            pdf.cell(0, 10, line + '.', 0, 1)
        pdf.output(self.pdf_path)

    def retrieve(self, query: str, k: int = 3) -> tuple[str, List[Dict]]:
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        relevant_docs = [d for d, score in docs if score < 1.4] 
        if not relevant_docs:
            return "", []
        context = "\n\n".join([d.page_content for d in relevant_docs])
        citations = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in relevant_docs]
        return context, citations

# Initialize RAG
rag_system = RAGSystem(pdf_path="/app/knowledge_base.pdf") # Use /app for Docker path

# ==========================================
# TASK 4: TOOLS
# ==========================================

search_tool = DuckDuckGoSearchRun()

def weather_tool_func(location: str) -> str:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        r = requests.get(url, params={"name": location, "count": 1}, timeout=5).json()
        if not r.get("results"): return f"Location '{location}' not found."
        lat, lon = r["results"][0]["latitude"], r["results"][0]["longitude"]
        w_url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "current_weather": True}
        w_res = requests.get(w_url, params=params).json()
        temp = w_res['current_weather']['temperature']
        return f"Current temperature in {location}: {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {e}"

def calculator_tool_func(expression: str) -> str:
    try:
        if not re.match(r"^[0-9+\-*/().\s]+$", expression):
            return "Invalid characters."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation error: {e}"

# ==========================================
# TASK 1 & 5: STATE & GRAPH
# ==========================================

class AgentState(TypedDict):
    query: str
    plan: str
    reasoning_trace: List[str]
    retrieved_context: str
    citations: List[Dict]
    tool_output: str
    final_answer: str

def planner_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    plan = "UNKNOWN"
    thought = f"Analyzing: '{state['query']}'. "
    
    if "weather" in query or "temperature" in query:
        plan = "use_tool"
        state["reasoning_trace"].append(thought + "Intent: Weather. Action: Call Weather Tool.")
    elif "calculate" in query or re.search(r"\d+[\+\-\*\/]\d+", query):
        plan = "use_tool"
        state["reasoning_trace"].append(thought + "Intent: Math. Action: Call Calculator.")
    elif "search" in query or "latest" in query:
        plan = "use_tool"
        state["reasoning_trace"].append(thought + "Intent: External Info. Action: Call Search Tool.")
    else:
        plan = "retrieve_rag"
        state["reasoning_trace"].append(thought + "Intent: Domain Query. Action: Retrieve from PDF.")
        
    state["plan"] = plan
    return state

def rag_retriever_node(state: AgentState) -> AgentState:
    state["reasoning_trace"].append("Executing RAG...")
    context, citations = rag_system.retrieve(state["query"])
    
    if not context:
        state["reasoning_trace"].append("RAG failed. Fallback to Search.")
        state["plan"] = "use_tool"
    else:
        state["retrieved_context"] = context
        state["citations"] = citations
        state["reasoning_trace"].append(f"RAG Success. Retrieved {len(citations)} chunks.")
    return state

def tool_executor_node(state: AgentState) -> AgentState:
    q = state["query"]
    result = ""
    
    if "weather" in q.lower():
        q_clean = q.rstrip("?,!. ")
        loc = "Unknown"
        if " in " in q_clean: loc = q_clean.split(" in ")[1]
        elif " at " in q_clean: loc = q_clean.split(" at ")[1]
        elif "weather" in q_clean: loc = q_clean.split("weather")[1]
        
        loc = loc.strip()
        result = weather_tool_func(loc)
        state["reasoning_trace"].append(f"Tool Used: Weather(loc={loc})")
        
    elif "calculate" in q.lower():
        expr = re.sub(r"[^\d+\-*/().]", "", q)
        result = calculator_tool_func(expr)
        state["reasoning_trace"].append(f"Tool Used: Calculator(expr={expr})")
        
    else:
        result = search_tool.run(q)
        state["reasoning_trace"].append("Tool Used: DuckDuckgo Search")
        
    state["tool_output"] = result
    return state

def synthesizer_node(state: AgentState) -> AgentState:
    state["reasoning_trace"].append("Synthesizing final answer...")
    
    # Logic for simple synthesis (or use OpenAI if key exists)
    if state.get("retrieved_context"):
        state["final_answer"] = f"Based on documents: {state['retrieved_context'][:300]}..."
    elif state.get("tool_output"):
        state["final_answer"] = f"Based on tools: {state['tool_output']}"
    else:
        state["final_answer"] = "I couldn't find an answer."
    return state

def decide_route(state: AgentState) -> str:
    return "tool_executor" if state["plan"] == "use_tool" else "rag_retriever"

def rag_decision(state: AgentState) -> str:
    return "tool_executor" if state["plan"] == "use_tool" else "synthesizer"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("rag_retriever", rag_retriever_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", decide_route, {"rag_retriever": "rag_retriever", "tool_executor": "tool_executor"})
workflow.add_conditional_edges("rag_retriever", rag_decision, {"tool_executor": "tool_executor", "synthesizer": "synthesizer"})
workflow.add_edge("tool_executor", "synthesizer")
workflow.add_edge("synthesizer", END)
app = workflow.compile()

# ==========================================
# TASK 6: UI
# ==========================================

def run_query(user_query: str, history: List):
    if not user_query.strip(): return history, "", ""
    inputs = {
        "query": user_query, "plan": "", "reasoning_trace": [],
        "retrieved_context": "", "citations": [], "tool_output": "", "final_answer": ""
    }
    result = app.invoke(inputs)
    
    response_text = f"**Answer:**\n{result['final_answer']}\n\n"
    citations_text = ""
    if result.get("citations"):
        citations_text = "**Sources:**\n" + "\n".join([f"- {c['source']} (p. {c['page']})" for c in result["citations"]])
    
    reasoning_text = "**ReAct Trace:**\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(result['reasoning_trace'])])
    history.append((user_query, response_text + citations_text))
    return history, reasoning_text, ""

with gr.Blocks(title="Cloud Multi-Agent System") as demo:
    gr.Markdown("# ☁️ Cloud Multi-Agent RAG")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            user_input = gr.Textbox(label="Question")
            with gr.Row():
                submit_btn = gr.Button("Ask")
        with gr.Column(scale=1):
            trace_box = gr.Textbox(label="ReAct Steps", lines=20, interactive=False)
    submit_btn.click(fn=run_query, inputs=[user_input, chatbot], outputs=[chatbot, trace_box, user_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
