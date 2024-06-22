from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus
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
], return_each_line=True).split_text(md)

vector_db = Milvus.from_documents(
    documents,
    embeddings,
    connection_args={'uri': MILVUS_URI}
)
