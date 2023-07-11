import openai
import pandas as pd
import os
from openai.embeddings_utils import cosine_similarity

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer


openai.api_key = os.environ.get('OPENAI_API_KEY')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def generate_knowledge(pdf_file):
    pdf_file = PyPDFLoader(pdf_file)
    pages = pdf_file.load_and_split()

    split = CharacterTextSplitter(chunk_size=400, separator = '  \n')
    texts = split.split_documents(pages)
    texts = [str(text.page_content) for text in texts] #Lista de parrafos
    paragraphs = pd.DataFrame(texts, columns=["text"])

    
    paragraphs['Embedding'] = paragraphs["text"].apply(lambda x: model.encode(x)) # Nueva columna con los embeddings de los parrafos
    paragraphs.to_csv('embeddings.csv')


pdf_file = "./juezBot/embedding/codigoPenal.pdf"
#generate_knowledge(pdf_file)

def get_knowledge(csv_file):
    knowledge_base = pd.read_csv(csv_file)
    knowledge_base = knowledge_base.drop(knowledge_base.columns[0], axis=1)
    knowledge_base['Embedding'] = knowledge_base['Embedding'].str.strip('[]').str.split()
    knowledge_base['Embedding'] = knowledge_base['Embedding'].apply(lambda x: [float(num) for num in x])
    return knowledge_base


csv_file = 'embeddings.csv'
knowledge_base = get_knowledge(csv_file)

# Consulta a la base de conocimiento

def get_context(question, knowledge_base):
    question_embed = model.encode(question)

    knowledge_base['Similarity'] = knowledge_base["Embedding"].apply(lambda x: cosine_similarity(x, question_embed))
    knowledge_base = knowledge_base.sort_values('Similarity', ascending=False)

    select_paragraphs = knowledge_base.head(3)
    context = ' '.join(select_paragraphs['text'])

    return context


question = '¿cual es la pena privativa por homicidio calificado?'
context = get_context(question, knowledge_base)

def get_answer(context, question, engine="text-davinci-002", temp=0.5, max_tokens=100, top_p=1, frequency_penalty=0, presence_penalty=0):
    prompt = "en base al siguiente contexto: "+context+"\nresponde a la siguiente pregunta: "+question
    answer = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return answer['choices'][0]['text']

answer = get_answer(context, question)

"""while True:
    question = input('Human : ')
    context = get_context(question=question, knowledge_base=knowledge_base)
    answer = get_answer(context, question)
    print('Bot: ', answer)"""