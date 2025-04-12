## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

1. Parsing and indexing the content of a PDF document.
2. Allowing users to ask questions in natural language.
3. Providing concise and accurate answers based on the content of the document.
  
The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

### DESIGN STEPS:

#### STEP 1: Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.


### PROGRAM:
```py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings = OpenAIEmbeddings()

    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 
```

### OUTPUT:

![exp-3 op](https://github.com/user-attachments/assets/d538c9da-7734-4b0b-b738-7799c4d3274d)

![exp-3 op2](https://github.com/user-attachments/assets/30e5bad2-2f7b-49b2-a1f7-1b80821b60b8)

![exp-3 op3](https://github.com/user-attachments/assets/c1dcbabf-12a4-4684-be38-bef61ad01869)


### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
