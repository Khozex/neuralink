import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Caminho para salvar e carregar o modelo
model_path = 'sentiment_model.h5'

# Carregar o dataset
df = pd.read_csv('data/Tweets.csv')

# Visualizar as primeiras linhas do dataset
print(df.head())

# Resumo estatístico do dataset
print(df.describe())

# Função para limpar o texto
def clean_text(text):
    return text.lower()

# Aplicar a limpeza de texto
df['text'] = df['text'].apply(clean_text)

# Transformar o sentimento em valores numéricos
le = LabelEncoder()
df['airline_sentiment'] = le.fit_transform(df['airline_sentiment'])

# Separar features (X) e rótulos (y)
X = df['text'].values
y = df['airline_sentiment'].values

# Transformar os textos em vetores numéricos usando TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar se o modelo já foi treinado e salvo anteriormente
if os.path.exists(model_path):
    # Carregar o modelo salvo
    model = load_model(model_path)
else:
    # Construir a rede neural
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Três classes para sentimentos: positivo, negativo, neutro

    # Compilar o modelo
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Salvar o modelo treinado
    model.save(model_path)

# Avaliar o desempenho do modelo
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))
