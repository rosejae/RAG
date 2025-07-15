from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from dotenv import load_dotenv
load_dotenv()

#
# chunking
#

loaders = [
    PyPDFLoader(r".\거대 언어 모델의 한국 이해도.pdf"),
    PyPDFLoader(r".\대규모 언어모델의 한국어 이해 능력 평가 방법에 관한 연구.pdf")
] 

docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10
)

splits = text_splitter.split_documents(docs)
 
#
# FAISS
#
    
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 13})    
    
#
# inference without reranking
#

llm = ChatOpenAI(model="gpt-4o")
qa1 = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    )

query = "대규모 언어모델의 한국어 이해 능력 평가 방법에는 무엇무엇이 있어?"
print(qa1.run(query=query))

#
# cohere (reranking)
#

compressor = CohereRerank(model="rerank-v3.5")
# ContextualCompressionRetriever: 관련 문서 검색
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever,
    )

compressed_docs = compression_retriever.get_relevant_documents(query)
print(compressed_docs)

qa2 = RetrievalQA.from_chain_type(
    llm=llm,                                 
    chain_type="stuff",                                 
    retriever=compression_retriever,
    )

print(qa2.run(query=query))