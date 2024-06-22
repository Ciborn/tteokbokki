from langchain.chains.flare.base import FlareChain
from langchain.globals import set_verbose
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus

from ..constants import EMBEDDINGS_MODEL, MILVUS_URI

set_verbose(True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model='llama3')

vector_db = Milvus(embeddings, connection_args={'uri': MILVUS_URI})
retriever = vector_db.as_retriever(search_kwargs={'k': 2})

# Translated and adapted from https://smith.langchain.com/hub/rlm/rag-prompt
prompt_template = '''
    Tu es un assistant qui répond à des questions. Utilise le contexte, qui représente des morceaux du rapport d'alternance de Robin BOURACHOT, pour répondre à la question posée. Ta réponse doit être complète, exhaustive, formulée en français, et doit être constituée d'au moins 3 phrases. Lorsque tu as fini de répondre, termine par "FINISHED".
    Question : {user_input}
    Contexte : {context}
    Réponse : {response}\
'''
prompt = ChatPromptTemplate.from_template(prompt_template)

question_generator_prompt_template = '''
    À l'aide d'un contexte partiel qui représente un élément de réponse à la question d'origine, formule une question dont la réponse est donnée.
    Question d'origine : {user_input}
    Contexte : {current_response}
    La question dont la réponse est "{uncertain_span}" est :
'''

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

def trace(x):
    print(x)

    return x

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
    min_prob=0.3,
)

output = flare.invoke('Qui travaille sur l\'entrepôt de données et dans quel cadre ?')
print(output)
