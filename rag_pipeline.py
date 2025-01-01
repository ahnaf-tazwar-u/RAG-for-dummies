import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import shutil
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import gradio as gr


#load database
DATA_PATH = 'dataset/cse299data_collect.txt'

def load_documents():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    return documents

#splitting text

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[15]
    # print("Sample chunk:", document.page_content)
    # print("Sample metadata:", document.metadata)

    return chunks

#save to chroma & print embadding

CHROMA_PATH = "chroma"


def save_to_chroma(chunks: list[Document]):
    # Remove existing directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Use SentenceTransformerEmbeddings
    db = Chroma.from_documents(
        chunks,
        embedding=OllamaEmbeddings(
            base_url='http://localhost:11434',
            model='all-minilm'
        ),
        persist_directory=CHROMA_PATH  # automatically persists in the defined path
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    # Print out the embeddings for verification
    # for chunk in chunks[:5]:  # Limit to 5 chunks for display
    #     embedding = db.embeddings.embed_documents([chunk.page_content])
    #     print("Embedding for chunk:", embedding)

#main logic

documents = load_documents()

chunks = split_text(documents)

save_to_chroma(chunks)

#query

# Load the Chroma database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(
    base_url='http://localhost:11434',  # Assuming Ollama server is running locally
    model='all-minilm'
))

# Create a retriever from the database
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize the Ollama LLM
ollama_llm = OllamaLLM(
    base_url='http://localhost:11434',
    model='llama3.2'  # Or any other supported model on your local Ollama server
)

# Create a RetrievalQA chain with Ollama as the LLM
qa = RetrievalQA.from_chain_type(llm=ollama_llm, chain_type="stuff", retriever=retriever)

# Define a prompt instruction for focused answers
PROMPT_INSTRUCTION = "Answer the question based on the provided context. Answer concisely and to the point."

# while True:
#     # Get user input for query
#     query = input("Please enter your query (or type 'exit' to quit): ")
#     if query.lower() == "exit":
#         print("Exiting the program.")
#         break

#     # Combine the prompt instruction with the user's query
#     full_query = f"{PROMPT_INSTRUCTION}\n\nQuestion: {query}"

#     Retrieve relevant chunks and combine them
#     retrieval_results = retriever.invoke(query)
#     combined_context = "\n".join([doc.page_content for doc in retrieval_results[:2]])

#     # Generate the answer using Ollama LLM
#     result = qa.invoke(full_query)

#     print("\nFinal Answer:")
#     print(result['result'])

# Function to process a single query
def process_query(chat_history, query):
    if query.lower() == "exit":
        chat_history.append({"role": "assistant", "content": "Exiting the program. Goodbye!"})
        return chat_history, ""
    
    try:
        # Combine the prompt instruction with the query
        full_query = f"{PROMPT_INSTRUCTION}\n\nQuestion: {query}"
        
        # Retrieve relevant chunks and combine them
        retrieval_results = retriever.invoke(query)
        combined_context = "\n".join([doc.page_content for doc in retrieval_results[:2]])
        
        # Generate the answer
        answer = qa.invoke(full_query)  # Use invoke instead of run (updated for compatibility)
        #print(answer)
        clean_answer = answer['result'].strip().split("Helpful Answer:")[-1].strip()
        
        # Append the user query and bot response to the chat history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": clean_answer})
        return chat_history,""
    
    except Exception as e:
        error_message = f"An error occurred: {e}"
        chat_history.append(("Bot", error_message))
        return chat_history,""

# Gradio UI with chat-like interface
def continuous_query():
    # Interactive Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# North South University ECE Department ChatBot")
        gr.Markdown("Chat with the bot. Type 'exit' to terminate the conversation.")
        
        chatbot = gr.Chatbot(label="Chat History", type="messages")
        query_input = gr.Textbox(
            label="Your Query",
            placeholder="Enter your question here... Type 'exit' to quit."
        )
        
        def handle_query(chat_history, query):
            return process_query(chat_history, query)
        
        query_input.submit(handle_query, inputs=[chatbot, query_input], outputs=[chatbot, query_input])
    
    demo.launch(debug=False, share=False)

# Launch the Gradio app
if __name__ == "__main__":
    continuous_query()
