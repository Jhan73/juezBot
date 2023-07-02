import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sentence_transformers import SentenceTransformer
from openai.embeddings_utils import cosine_similarity

homicidios_df = pd.read_csv('./datasetHomicidios.csv', encoding='utf-8')
X = homicidios_df.drop(['CASOS','Pena'], axis=1)
Y = homicidios_df['Pena']

def data_cleansing(X):
    numerical = X['Edad del acusado']
    categoriacal = X.drop(['Edad del acusado'], axis=1)

    TipoHomicidio = {'Simple': 1, 'Calificado': 2}
    AntecedentesAcusado = {'ausencia de antecedentes': 1, 'Tiene antecedentes por delitos violentos': 2, 'ha cometido homicidios previos': 3}
    Agravantes = {'uso de arma blanca': 1, 'uso de armas de fuego': 2,'planificacion del crimen': 3,'participacion de multiples acusados': 4}
    EvidenciasPresentadas = {'evaluaciones psicologicas': 1, 'testimonios de testigos': 2, 'registros de comunicaciones': 3,'pruebas forenses': 4}
    RelacionAcusadoVictima = {'conocidos': 1,'parejas': 2, 'familiares': 3}
    MotivacionCrimen = {'defensa propia': 1, 'celos': 2, 'violencia domestica': 3,'venganza': 4}
    ConductaAcusado = {'Muestra arrepentimiento genuino': 1,'ha realizado acciones de reparacion': 2,'muestra remordimiento': 3,'colaboracion con la justicia': 4}
    CausalidadesExternas = {'presion de grupos delectivos': 1,'consumo de alcohol': 2,'consumo de drogas': 3}
    EvaluacionPsicologica = {'condiciones que afectan la capacidad de juicio': 1,'transtornos mentales': 2}
    GradoViolencia = {'tortura': 1, 'mutilacion': 2,'violencia extrema': 3}

    cat_numerical = pd.DataFrame()
    cat_numerical['Tipo de homicidio'] = categoriacal['Tipo de homicidio'].copy().map(TipoHomicidio)
    cat_numerical['Antecedentes del acusado'] = categoriacal['Antecedentes del acusado'].copy().map(AntecedentesAcusado)
    cat_numerical['Agravantes'] = categoriacal['Agravantes'].copy().map(Agravantes)
    cat_numerical['Evidencias presentadas en el juicio'] = categoriacal['Evidencias presentadas en el juicio'].copy().map(EvidenciasPresentadas)
    cat_numerical['Relacion entre el acusado y la victima'] = categoriacal['Relacion entre el acusado y la victima'].copy().map(RelacionAcusadoVictima)
    cat_numerical['Motivacion del crimen'] = categoriacal['Motivacion del crimen'].copy().map(MotivacionCrimen)
    cat_numerical['Conducta del acusado durante el juicio'] = categoriacal['Conducta del acusado durante el juicio'].copy().map(ConductaAcusado)
    cat_numerical['Causalidades externas'] = categoriacal['Causalidades externas'].copy().map(CausalidadesExternas)
    cat_numerical['Evaluacion psicologica'] = categoriacal['Evaluacion psicologica'].copy().map(EvaluacionPsicologica)
    cat_numerical['Grado de violencia'] = categoriacal['Grado de violencia'].copy().map(GradoViolencia)

    cat_numerical = cat_numerical.fillna(0)

    X = pd.concat([numerical, cat_numerical], axis=1)
    
    return X

X = data_cleansing(X)

def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rf_reg = RandomForestRegressor(random_state=42, n_estimators=500)
    regressor = rf_reg.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Rerror: ', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

    return sc, regressor

sc, regressor = train_model(X, Y)

def predict (New_data, sc = sc, regressor= regressor ):
    New_df = pd.DataFrame(New_data, index=[0])

    TipoHomicidio = {'Simple': 1, 'Calificado': 2}
    AntecedentesAcusado = {'ausencia de antecedentes': 1, 'Tiene antecedentes por delitos violentos': 2, 'ha cometido homicidios previos': 3}
    Agravantes = {'uso de arma blanca': 1, 'uso de armas de fuego': 2,'planificacion del crimen': 3,'participacion de multiples acusados': 4}
    EvidenciasPresentadas = {'evaluaciones psicologicas': 1, 'testimonios de testigos': 2, 'registros de comunicaciones': 3,'pruebas forenses': 4}
    RelacionAcusadoVictima = {'conocidos': 1,'parejas': 2, 'familiares': 3}
    MotivacionCrimen = {'defensa propia': 1, 'celos': 2, 'violencia domestica': 3,'venganza': 4}
    ConductaAcusado = {'Muestra arrepentimiento genuino': 1,'ha realizado acciones de reparacion': 2,'muestra remordimiento': 3,'colaboracion con la justicia': 4}
    CausalidadesExternas = {'presion de grupos delectivos': 1,'consumo de alcohol': 2,'consumo de drogas': 3}
    EvaluacionPsicologica = {'condiciones que afectan la capacidad de juicio': 1,'transtornos mentales': 2}
    GradoViolencia = {'tortura': 1, 'mutilacion': 2,'violencia extrema': 3}

    New_df['Tipo de homicidio'] = New_df['Tipo de homicidio'].copy().map(TipoHomicidio)
    New_df['Antecedentes del acusado'] = New_df['Antecedentes del acusado'].copy().map(AntecedentesAcusado)
    New_df['Agravantes'] = New_df['Agravantes'].copy().map(Agravantes)
    New_df['Evidencias presentadas en el juicio'] = New_df['Evidencias presentadas en el juicio'].copy().map(EvidenciasPresentadas)
    New_df['Relacion entre el acusado y la victima'] = New_df['Relacion entre el acusado y la victima'].copy().map(RelacionAcusadoVictima)
    New_df['Motivacion del crimen'] = New_df['Motivacion del crimen'].copy().map(MotivacionCrimen)
    New_df['Conducta del acusado durante el juicio'] = New_df['Conducta del acusado durante el juicio'].copy().map(ConductaAcusado)
    New_df['Causalidades externas'] = New_df['Causalidades externas'].copy().map(CausalidadesExternas)
    New_df['Evaluacion psicologica'] = New_df['Evaluacion psicologica'].copy().map(EvaluacionPsicologica)
    New_df['Grado de violencia'] = New_df['Grado de violencia'].copy().map(GradoViolencia)

    New_df = New_df.fillna(0)

    New_df_scaled = sc.transform(New_df)
    prediction = regressor.predict(New_df_scaled)

    return prediction

data = pd.read_csv('vars.csv')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
def format_data(description, data = data):
    data['Embedding'] = data["Variables"].apply(lambda x: model.encode(x))
    description_embed = model.encode(description)
    data['Similarity'] = data["Embedding"].apply(lambda x: cosine_similarity(x, description_embed))
    data = data.sort_values('Similarity', ascending=False)

    case_data = {'Edad del acusado': 0,'Tipo de homicidio':'', 'Antecedentes del acusado':'', 'Agravantes':'',
       'Evidencias presentadas en el juicio':'','Relacion entre el acusado y la victima':'', 'Motivacion del crimen':'',
       'Conducta del acusado durante el juicio':'', 'Causalidades externas':'','Evaluacion psicologica':'', 'Grado de violencia':''}
    
    selected_data =  data.head()
    case_data[selected_data.iloc[0,0]] = selected_data.iloc[0,1]
    case_data[selected_data.iloc[1,0]] = selected_data.iloc[1,1]
    case_data[selected_data.iloc[2,0]] = selected_data.iloc[2,1]
    case_data[selected_data.iloc[3,0]] = selected_data.iloc[3,1]
    case_data[selected_data.iloc[4,0]] = selected_data.iloc[4,1]

    case_df = pd.DataFrame(case_data, index=[0])

    return case_data, case_df

description = 'Una persona es acusada por haber cometido homicidio calificado, el crimen fue ejecutado con complice con otras personas, ademas para el caso se presentaron testimonios que corroboran el hecho'

case_data, _ = format_data(description)
prediction = predict(case_data)

print("Prediccion de la pena privativa: ", prediction)














