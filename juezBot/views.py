import os
from django.http import HttpResponse
from django.template import loader

import pandas as pd
import openai
from sentence_transformers import SentenceTransformer


#from embedding.embedding import knowledge_base, get_context, get_answer
#from modelo.randomForest import homicidios_df, Y, data, format_data, predict

openai.api_key = os.environ.get('OPENAI_API_KEY')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

"""while True:
    prompt = input('Human : ')

    if (prompt[0] == 'D'):
        context = get_context(prompt, knowledge_base)
        answer = get_answer(context, prompt)
    elif (prompt[0] == 'M'):
        case_df = format_data(prompt)
        prediction = predict(case_df)
        answer = f'La pena privativa de libertad para el caso es de {prediction} años'
    else:
        engine="text-davinci-002"
        temp=0.5
        max_tokens=100
        top_p=1
        frequency_penalty=0
        presence_penalty=0
        answer = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        answer = answer['choices'][0]['text']
    print(f'Bot: {answer}')"""

def saludo(request):
    t = loader.get_template("index.html")
    c = {"saludo": 'Hola mundo!!'}
    return HttpResponse(t.render(c, request), content_type = "application/xhtml+xml")
