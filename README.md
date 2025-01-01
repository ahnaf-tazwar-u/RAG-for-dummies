# Retrieval-Augmented Generation (RAG) Pipeline

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to enhance the capabilities of Large Language Models (LLMs) by integrating external data sources. The pipeline retrieves relevant documents and augments LLM responses with context-aware, precise information.

This project was completed as part of the *CSE299: Junior Design Project* course at North South University.

---

## Why RAG?
While LLMs like GPT are powerful, their knowledge is limited to their training data and cutoff date. RAG bridges this gap by enabling models to reason about new or private data. This process combines retrieval and generation to produce responses grounded in external knowledge.

---

## Features
- **Data Retrieval**: Use external data for improved context.  
- **Context-Aware Responses**: Generate precise answers based on retrieved documents.  
- **Modular Design**: Seamlessly integrate tools like LangChain, Chroma, and Ollama.  
- **User-Friendly Interface**: Interact with the pipeline using Gradio.

---

## Libraries and Requirements
The following dependencies are required for the project:

chromadb==0.5.18  
langchain-community==0.3.7  
langchain-ollama==0.2.0  
langchain-chroma==0.1.4  
gradio==5.6.0  

---

## How to Use

### Option 1: Run in Google Colab
1. Download the repository files.
2. Upload the `rag_pipeline.ipynb` notebook and required documents to Google Colab.
3. Install the required libraries directly in Colab.
4. Execute the notebook cells to run the pipeline.

### Option 2: Run Locally

#### Prerequisites
Ensure the following are installed:
- Python 3.9 or later.  
- Git.  
- Docker Desktop.

#### Steps to Set Up
1. Clone the repository.
2. Set up a virtual environment and activate it.  
4. Install the dependencies using the `requirements.txt` file.  
5. Install and configure Docker for Ollama: enable CPU virtualization via BIOS/UEFI and run the Ollama Docker model.  
6. Run the pipeline locally by executing the script or opening the notebook.

---

## Pipeline Workflow
1. **Data Loading**: Load external documents into the pipeline.  
2. **Chunking**: Split documents into smaller, meaningful parts.  
3. **Vectorization**: Convert text chunks into embeddings using Chroma.  
4. **Querying and Retrieval**: Match user queries with stored embeddings.  
5. **Response Generation**: Use Ollama's LLM to generate responses based on retrieved data.

---

## Contributors
This project was developed as part of *CSE299: Junior Design Project* at North South University. Contributors include:
- Ahnaf Tazwar Unmesh  
- Faisal Bin Zaman
