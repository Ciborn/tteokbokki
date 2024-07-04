from langchain.chains.flare.base import FlareChain
from langchain.globals import set_verbose
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.storage import RedisStore
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..constants import EMBEDDINGS_MODEL, MILVUS_URI

set_verbose(True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model='llama3:8b')

child_splitter = RecursiveCharacterTextSplitter(separators=['\n'])
parent_splitter = RecursiveCharacterTextSplitter(separators=['###', '####'])

vector_db = Milvus(embeddings, connection_args={'uri': MILVUS_URI})

store = RedisStore(redis_url='redis://localhost:6379')
retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    byte_store=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1},
)

# Translated and adapted from https://smith.langchain.com/hub/rlm/rag-prompt
prompt_template = '''
    Tu es un assistant qui répond à des questions. Utilise le contexte, qui représente des morceaux du rapport d'alternance de Robin BOURACHOT, pour répondre à la question posée. Ta réponse doit être complète, exhaustive, formulée en français, et doit être constituée d'au moins 3 phrases. Lorsque tu as fini de répondre, termine par "FINISHED".
    >>> QUESTION PRINCIPALE : {user_input}
    >>> CONTEXTE : {context}
    >>> RÉPONSE : {response}\
'''
prompt = ChatPromptTemplate.from_template(prompt_template)

question_generator_prompt_template = '''
    À l'aide d'une réponse partielle qui représente un élément de réponse à la question d'origine, formule une question dont la réponse est donnée.
    >>> QUESTION D'ORIGINE : {user_input}
    >>> RÉPONSE PRÉCÉDENTE PARTIELLE : {current_response}
    >>> Formule une question à laquelle on peut répondre par "{uncertain_span}" :
'''

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["user_input", "context", "response"])

question_generator_prompt = PromptTemplate(
    template=question_generator_prompt_template,
    input_variables=["user_input", "current_response", "uncertain_span"])

flare = FlareChain.from_llm(
    llm,
    retriever=retriever,
    prompt=prompt,
    question_generator_prompt=question_generator_prompt,
    max_generation_len=164,
    min_prob=0.25,
)

output = flare.invoke('Quels sont les projets en cours à l\'institut ?')
print(output)
