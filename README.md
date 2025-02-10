
# Document-Based Question Answering System Using RAG

This project implements a **Document-Based Question Answering (QA) System** that uses **Retrieval-Augmented Generation (RAG)** to answer queries from PDF documents. It utilizes **FAISS** for vector search, **Hugging Face embeddings** for text representation, and a **quantized LLaMA-2 model** for response generation. The system is deployed with **Chainlit** to provide an interactive chat interface for real-time user interactions.

---

## **Technology Stack Used**

- **Language & Frameworks:** Python, LangChain, Chainlit
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings Model:** Hugging Face (sentence-transformers/all-MiniLM-L6-v2)
- **LLM (Language Model):** TheBloke/Llama-2-7B-Chat-GGML (Quantized) via CTransformers
- **Document Processing:** PyPDFLoader, DirectoryLoader
- **Prompt Engineering:** Custom Prompt Template with LangChain
- **Retrieval-Augmented Generation (RAG):** For context-based question answering
- **Deployment & Interface:** Chainlit (for chatbot UI)

---

## **Project Overview**

This project is designed to answer user queries based on the content of PDF documents. It preprocesses PDF files into text, splits the text into chunks, and generates embeddings for efficient search and retrieval. When a query is made, the system retrieves the most relevant document chunks, passes them to a **quantized LLaMA-2 model** for answer generation, and presents the result through an interactive **Chainlit** chatbot interface.

---

## **Features**

- **Document Preprocessing:** Load and split PDF files into smaller chunks for better retrieval accuracy.
- **Vector Search:** Use FAISS to store and search document embeddings for relevant context.
- **RAG Approach:** Retrieve the most relevant document chunks before generating the final response using LLaMA-2.
- **Interactive Chat Interface:** Use Chainlit for a seamless user experience where users can ask questions and receive answers in real time.
- **Efficient Querying:** Answers are based on the context retrieved from the documents, ensuring higher accuracy.

---

## **Setup Instructions**

### **Prerequisites**

- Python 3.7 or higher
- Install the necessary libraries and dependencies:

```bash
pip install -r requirements.txt
```

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/document-qa-system.git
cd document-qa-system
```

### **2. Install Dependencies**

Make sure to have all dependencies installed by running the following command:

```bash
pip install -r requirements.txt
```

### **3. Preparing the Vector Database**

- Store your PDF documents in the `data/` directory.
- Run the script `ingest.py` to preprocess and create embeddings for the documents:

```bash
python ingest.py
```

This will create a FAISS vector database in the `vectorstore/db_faiss` directory.

### **4. Running the Chat Interface**

Run the chatbot interface using **Chainlit**:

```bash
chainlit run model.py -w
```

This will start the Chainlit interface, and you can interact with the bot in real-time via the browser.

---

## **Usage**

1. Start the chatbot by running `model.py`.
2. Once the system is running, visit the provided URL to interact with the bot.
3. Ask questions related to the content of the documents, and the system will generate context-based answers.

---

## **How It Works**

1. **Document Loading & Preprocessing:** 
   - PDF files from the `data/` directory are loaded and split into smaller chunks using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
   
2. **Creating Embeddings & Vector Database:** 
   - The text chunks are converted into embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model and stored in a FAISS vector database.

3. **Retrieval-Augmented Generation (RAG):**
   - Upon a user query, the most relevant document chunks are retrieved from the vector database and passed to the **LLaMA-2 model** for generating a response.

4. **Interactive Chat with Chainlit:**
   - The chatbot interface is powered by **Chainlit**, allowing real-time interaction with users. The user can ask questions, and the system will provide accurate answers based on the documents.

---

## **Contributing**

Feel free to fork the repository, make changes, and submit pull requests. Contributions to enhance the accuracy of the system or improve the user interface are welcome.

---
## **Acknowledgments**

- **LangChain** for providing modular tools for document processing and LLM integration.
- **FAISS** for efficient vector search.
- **Chainlit** for providing the easy-to-use chat interface.

---
