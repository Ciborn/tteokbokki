from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus

from ..constants import EMBEDDINGS_MODEL, MILVUS_URI

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model='llama3:8b')

vector_db = Milvus(embeddings, connection_args={'uri': MILVUS_URI})
retriever = vector_db.as_retriever()

# Translated and adapted from https://smith.langchain.com/hub/rlm/rag-prompt
prompt_template = '''
    Tu es un assistant qui répond à des questions. Utilise le contexte, qui représente des morceaux du rapport d'alternance de Robin BOURACHOT, pour répondre à la question posée. Ta réponse doit être complète, exhaustive, formulée en français, et doit être constituée d'au moins 3 phrases.
    QUESTION : {question}
    CONTEXTE : {context}
    RÉPONSE : 
'''
prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    { 'context': retriever | format_docs, 'question': RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)

output = rag_chain.invoke('Quels sont les projets en cours à l\'institut ?')
print(output)
