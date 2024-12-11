import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import googleapiclient.discovery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

# Configuración para suprimir warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

class YouTubeVideoClassifier:
    def __init__(self, categories):
        self.categories = categories
        self.label_encoder = LabelEncoder()
        self.model = None
        self.vectorizador_texto = TfidfVectorizer(max_features=1000)
        
        # Configuración de API de YouTube
        API_KEY = 'AIzaSyCwCKa-nN7ea_JHiqLalg9AO5jHqhvUPSs'
        self.youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)

    def obtener_videos_por_categoria(self, query, max_videos=200):
        videos_data = []
        
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_videos,
                order="relevance",  # Puedes cambiar entre: relevance, rating, viewCount, date
                relevanceLanguage="es"  # Opcional: filtrar por idioma
            )
            
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                titulo = item['snippet']['title']
                descripcion = item['snippet']['description']
                miniatura_url = item['snippet']['thumbnails']['high']['url']
                
                try:
                    # Descargar miniatura
                    response = requests.get(miniatura_url)
                    miniatura = Image.open(BytesIO(response.content))
                    
                    # Procesar miniatura
                    miniatura_procesada = self.procesar_miniatura(miniatura)
                    
                    videos_data.append({
                        'video_id': video_id,
                        'titulo': titulo,
                        'descripcion': descripcion,
                        'miniatura': miniatura,  # Guardar imagen original
                        'miniatura_procesada': miniatura_procesada,
                        'categoria': query
                    })
                    
                except Exception as e:
                    print(f"Error procesando video {video_id}: {e}")
        
        except Exception as e:
            print(f"Error en la búsqueda de YouTube: {e}")
        
        return videos_data

    def procesar_miniatura(self, miniatura):
        """
        Procesar miniatura para extraer características
        """
        # Redimensionar
        miniatura_resized = miniatura.resize((224, 224))
        
        # Convertir a escala de grises
        miniatura_gray = miniatura_resized.convert('L')
        
        # Convertir a array numpy
        miniatura_array = np.array(miniatura_gray).flatten()
        
        return miniatura_array

    def preparar_datos(self, datos_videos):
        """
        Preparar datos para entrenamiento
        """
        # Características de miniaturas
        X_imagenes = np.array([dato['miniatura_procesada'] for dato in datos_videos])
        
        # Características de texto (título y descripción)
        textos = [f"{dato['titulo']} {dato['descripcion']}" for dato in datos_videos]
        
        # Vectorizar texto
        X_texto = self.vectorizador_texto.fit_transform(textos).toarray()
        
        # Codificar etiquetas
        y = self.label_encoder.fit_transform([dato['categoria'] for dato in datos_videos])
        
        return X_imagenes, X_texto, y, datos_videos

    def crear_modelo_hibrido(self, input_shape_img, input_shape_texto):
        """
        Crear modelo híbrido que combina características de imagen y texto
        """
        # Entrada para imágenes
        input_img = tf.keras.layers.Input(shape=(input_shape_img,), name='input_img')
        
        # Entrada para texto
        input_texto = tf.keras.layers.Input(shape=(input_shape_texto,), name='input_texto')
        
        # Capas para imágenes
        x_img = tf.keras.layers.Dense(128, activation='relu')(input_img)
        x_img = tf.keras.layers.Dropout(0.5)(x_img)
        
        # Capas para texto
        x_texto = tf.keras.layers.Dense(64, activation='relu')(input_texto)
        x_texto = tf.keras.layers.Dropout(0.5)(x_texto)
        
        # Concatenar características
        x = tf.keras.layers.Concatenate()([x_img, x_texto])
        
        # Capas finales
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(len(self.categories), activation='softmax')(x)
        
        # Crear modelo
        modelo = tf.keras.Model(inputs=[input_img, input_texto], outputs=output)
        
        # Compilar modelo
        modelo.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return modelo

    def entrenar_modelo(self, X_train_img, X_train_texto, y_train, X_test_img, X_test_texto, y_test):
        """
        Entrenar modelo híbrido
        """
        # Crear modelo con dimensiones dinámicas
        modelo = self.crear_modelo_hibrido(X_train_img.shape[1], X_train_texto.shape[1])
        
        # Entrenar modelo
        historia = modelo.fit(
            [X_train_img, X_train_texto], y_train, 
            validation_data=([X_test_img, X_test_texto], y_test),
            epochs=10, 
            batch_size=32
        )
        
        # Guardar modelo
        self.model = modelo
        
        return historia

    def evaluar_modelo(self, X_test_img, X_test_texto, y_test):
        """
        Evaluar rendimiento del modelo
        """
        return self.model.evaluate([X_test_img, X_test_texto], y_test)

    def visualizar_resultados(self, datos_videos, predicciones):
        """
        Visualizar resultados de clasificación de forma aleatoria
        """
        plt.figure(figsize=(20, 10))
        
        # Seleccionar aleatoriamente 12 videos de los disponibles
        muestras_aleatorias = random.sample(list(zip(datos_videos, predicciones)), min(12, len(datos_videos)))
        
        for i, (dato, prediccion) in enumerate(muestras_aleatorias, 1):
            plt.subplot(3, 4, i)
            
            # Mostrar miniatura
            plt.imshow(dato['miniatura'])
            
            # Título del video
            titulo_corto = dato['titulo'][:30] + '...' if len(dato['titulo']) > 30 else dato['titulo']
            
            # Categoría predicha
            categoria_predicha = self.categories[prediccion]
            
            plt.title(f"Categoría: {categoria_predicha}\n{titulo_corto}", fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle("Clasificación de Videos de YouTube", fontsize=16)
        plt.show()

def main():
    random.seed(time.time())
    # Configuración de categorías
    categorias = ['juegos_accion', 'juegos_estrategia', 'juegos_competitivos']
    
    clasificador = YouTubeVideoClassifier(categorias)
    # Recolectar datos para cada categoría
    datos_videos = []
    for categoria in categorias:
        videos_categoria = clasificador.obtener_videos_por_categoria(categoria)
        
        # Mezclar aleatoriamente los videos de cada categoría
        random.shuffle(videos_categoria)
        
        # Tomar solo una porción aleatoria
        datos_videos.extend(videos_categoria[:100])  # Limitar a 100 por categoría
    
    # Verificar si hay datos suficientes
    if len(datos_videos) == 0:
        print("No se pudieron obtener videos. Verificar configuración.")
        return
    
    # Preparar datos
    X_imagenes, X_texto, y, datos_videos_originales = clasificador.preparar_datos(datos_videos)
    
    # Dividir datos
    X_train_img, X_test_img, X_train_texto, X_test_texto, y_train, y_test = train_test_split(
        X_imagenes, X_texto, y, test_size=0.2
    )
    
    # Entrenar modelo
    historia = clasificador.entrenar_modelo(X_train_img, X_train_texto, y_train, 
                                            X_test_img, X_test_texto, y_test)
    
    # Evaluar modelo
    resultado = clasificador.evaluar_modelo(X_test_img, X_test_texto, y_test)
    print(f"Precisión del modelo: {resultado[1]*100}%")
    
    # Hacer predicciones
    predicciones = clasificador.model.predict([X_test_img, X_test_texto])
    predicciones_clase = np.argmax(predicciones, axis=1)
    
    # Visualizar resultados
    clasificador.visualizar_resultados(
        [datos_videos_originales[i] for i in np.where(y_test != None)[0]], 
        predicciones_clase
    )
    
    # Crear matriz de confusión
    cm = confusion_matrix(y_test, predicciones_clase)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categorias, 
                yticklabels=categorias)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

    #Análisis por posición:

#Primera fila (juegos_accion):
#30: Correctamente clasificados como acción
#5: Clasificados incorrectamente como estrategia
#2: Clasificados incorrectamente como competitivos