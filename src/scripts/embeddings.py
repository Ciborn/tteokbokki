from langchain_community.storage import RedisStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

from ..constants import EMBEDDINGS_MODEL, MILVUS_URI

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

with open('./rapport.md', 'r') as f:
    md = f.read()

documents = MarkdownHeaderTextSplitter([
    ('#', 'grand_title'),
    ('##', 'title'),
    ('###', 'subtitle'),
    ('####', 'section_title'),
]).split_text(md)

child_splitter = RecursiveCharacterTextSplitter(separators=['\n'])
parent_splitter = RecursiveCharacterTextSplitter(separators=['###', '####'])

vector_db = Milvus(embeddings, connection_args={'uri': MILVUS_URI}, auto_id=True, drop_old=True)

store = RedisStore(redis_url='redis://localhost:6379')
retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    byte_store=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)
print(retriever.invoke('projets en cours'))
