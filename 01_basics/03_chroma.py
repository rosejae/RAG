import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

#
# load pdf
#

loaders = [
    PyPDFLoader(r".\거대 언어 모델의 한국 이해도.pdf"),
    PyPDFLoader(r".\대규모 언어모델의 한국어 이해 능력 평가 방법에 관한 연구.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

#
# chunking
#

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
    )

splits = text_splitter.split_documents(docs)
# print(splits)

#
# embedding model, vector database
#

embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002", 
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

persist_directory = r'.\chroma'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(f"numbers of documents (embedded): {vectordb._collection.count()}")

#
# inference
#

# 여기에는 LLM으로 추론화는 과정이 빠져있음
question = "한국형 LLM?"
docs = vectordb.similarity_search(question, k=3)  #return 받고자하는 문서의 수를 3개로 지정

# 문서 길이 확인
print(len(docs))

# 첫번째 문서의 콘텐츠 확인
print(docs[0].page_content)

# 영구 저장
vectordb.persist()