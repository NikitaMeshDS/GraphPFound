"""
Модуль для генерации матрицы выдачи товаров с вероятностями P_click, P_buy, P_look1, P_look2
"""

import numpy as np
import pandas as pd
import re
import os
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from load_data import load_all

# Инициализация NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    stopwords_set = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stopwords_set = set(stopwords.words('english'))

sno = SnowballStemmer('english')

def preprocess_query(query):
    """
    Предобработка текстового запроса пользователя
    """
    def decontracted(phrase):
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can't", "can not", phrase)
        phrase = re.sub(r"n't", " not", phrase)
        phrase = re.sub(r"'re", " are", phrase)
        phrase = re.sub(r"'s", " is", phrase)
        phrase = re.sub(r"'d", " would", phrase)
        phrase = re.sub(r"'ll", " will", phrase)
        phrase = re.sub(r"'t", " not", phrase)
        phrase = re.sub(r"'ve", " have", phrase)
        phrase = re.sub(r"'m", " am", phrase)
        return phrase
    
    def remove_pun(sentence):
        cleaned = re.sub(r'[?|!|\\|#|.|"|)|(|)|/|,|:|\'|-|$|+|~|;|-|_|@|>|<]', r'', sentence)
        return cleaned
    
    sentence = decontracted(query)
    sentence = re.sub(r"\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = " ".join((sno.stem(e.lower())) for e in sentence.split() if e.lower() not in stopwords_set)
    return remove_pun(sentence).strip()


def generate_feed_tfidf(query, n_products, vectorizer, product_vectors, data_df, rows=2):
    """
    Генерирует выдачу товаров на основе TF-IDF эмбеддингов
    
    Параметры:
    -----------
    query : str
        Текстовый запрос пользователя
    n_products : int
        Количество товаров для выдачи
    vectorizer : TfidfVectorizer
        Векторизатор TF-IDF
    product_vectors : sparse matrix
        TF-IDF векторы всех товаров
    data_df : pandas.DataFrame
        DataFrame с данными о товарах
    rows : int
        Количество строк в сетке (для форматирования)
    
    Возвращает:
    -----------
    results : list of dict
        Список товаров с полями: id, product_name, image_url, relevance_score, vector_index
    """
    query_processed = preprocess_query(query)
    query_vector = vectorizer.transform([query_processed])
    
    # Вычисляем схожесть запроса со всеми товарами
    similarities = cosine_similarity(query_vector, product_vectors)[0]
    
    # Нормализуем схожесть в [0, 1]
    min_sim = similarities.min()
    max_sim = similarities.max()
    if max_sim - min_sim > 0:
        relevance_scores = (similarities - min_sim) / (max_sim - min_sim)
    else:
        relevance_scores = np.ones_like(similarities)
    
    # Выбор топ-n товаров
    top_indices = np.argsort(relevance_scores)[::-1][:n_products]
    
    # Формирование результатов
    results = []
    for idx in top_indices:
        product_idx = data_df.index[idx]
        results.append({
            'id': data_df.loc[product_idx, 'asin'],
            'product_name': data_df.loc[product_idx, 'product_name'],
            'image_url': data_df.loc[product_idx, 'medium'],
            'relevance_score': float(relevance_scores[idx]),
            'vector_index': int(idx)
        })
    
    return results


# Глобальная переменная для кэширования модели эмбеддингов
_cached_embedding_model = None


def get_embedding_model(w2v_model=None):
    """Получает модель для эмбеддингов: Word2Vec или None (fallback на TF-IDF)"""
    global _cached_embedding_model
    
    if _cached_embedding_model is not None:
        return _cached_embedding_model
    
    if w2v_model is not None:
        if hasattr(w2v_model, 'vector_size') or hasattr(w2v_model, 'index_to_key') or hasattr(w2v_model, 'wv'):
            _cached_embedding_model = w2v_model
            return _cached_embedding_model
    
    try:
        from gensim.models import KeyedVectors
        w2v_path = '/kaggle/input/gensimmodel/GoogleNews-vectors-negative300.bin'
        if os.path.exists(w2v_path):
            _cached_embedding_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            return _cached_embedding_model
    except:
        pass
    
    return None


def calculate_purchase_probability_tfidf(query, product_name, tfidf_dict=None, tfidf_feat=None, data_df=None):
    """
    Fallback функция: вычисляет вероятность покупки используя только TF-IDF
    """
    if tfidf_dict is None or tfidf_feat is None:
        if data_df is not None:
            tifd_temp = TfidfVectorizer()
            tifd_temp.fit_transform(data_df['product_name'])
            tfidf_dict = dict(zip(tifd_temp.get_feature_names_out(), list(tifd_temp.idf_)))
            tfidf_feat = tifd_temp.get_feature_names_out()
        else:
            return 0.0, 0.0
    
    query_processed = preprocess_query(query)
    product_processed = preprocess_query(product_name)
    
    # Создаем простые TF-IDF векторы
    query_words = query_processed.split()
    product_words = product_processed.split()
    
    # Вычисляем TF-IDF вручную
    def get_tfidf_vector(words, tfidf_dict, tfidf_feat):
        vec = np.zeros(len(tfidf_feat))
        word_to_idx = {word: idx for idx, word in enumerate(tfidf_feat)}
        for word in words:
            if word in word_to_idx:
                tf = words.count(word) / len(words) if len(words) > 0 else 0
                idf = tfidf_dict.get(word, 1.0)
                vec[word_to_idx[word]] = tf * idf
        return vec
    
    query_vec = get_tfidf_vector(query_words, tfidf_dict, tfidf_feat)
    product_vec = get_tfidf_vector(product_words, tfidf_dict, tfidf_feat)
    
    similarity = cosine_similarity([query_vec], [product_vec])[0][0]
    probability = (similarity + 1) / 2
    
    return float(probability), float(similarity)


def calculate_purchase_probability(query, product_name, embedding_model=None, data_df=None, w2v_model=None):
    """Вычисляет вероятность покупки на основе схожести запроса и названия товара"""
    if embedding_model is None:
        embedding_model = get_embedding_model(w2v_model=w2v_model)
        if embedding_model is None:
            return calculate_purchase_probability_tfidf(query, product_name, None, None, data_df)
    
    try:
        query_processed = preprocess_query(query)
        product_processed = preprocess_query(product_name)
        
        def get_word2vec_embedding(text, model):
                words = text.split()
                if len(words) == 0:
                    # Определяем размерность вектора из модели
                    vec_size = 300
                    if hasattr(model, 'vector_size'):
                        vec_size = model.vector_size
                    elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
                        vec_size = model.wv.vector_size
                    return np.zeros(vec_size)
                
                embeddings = []
                for word in words:
                    try:
                        if hasattr(model, 'index_to_key') and word in model.index_to_key:
                            embeddings.append(model[word])
                        elif hasattr(model, 'wv') and word in model.wv:
                            embeddings.append(model.wv[word])
                        elif hasattr(model, '__getitem__'):
                            # Для моделей из gensim.downloader
                            embeddings.append(model[word])
                    except:
                        continue
                
                if len(embeddings) == 0:
                    vec_size = 300
                    if hasattr(model, 'vector_size'):
                        vec_size = model.vector_size
                    elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
                        vec_size = model.wv.vector_size
                    return np.zeros(vec_size)
                return np.mean(embeddings, axis=0)
        
        query_embedding = get_word2vec_embedding(query_processed, embedding_model)
        product_embedding = get_word2vec_embedding(product_processed, embedding_model)
        
        similarity = cosine_similarity([query_embedding], [product_embedding])[0][0]
        probability = (similarity + 1) / 2
        
        return float(probability), float(similarity)
    except:
        return calculate_purchase_probability_tfidf(query, product_name, None, None, data_df)


def generate_feed_matrix(query, n_products, vectorizer, product_vectors, data_df, rows=2, cols=2, embedding_model=None, image_model=None, w2v_model=None):
    """
    Генерирует матрицу размера n * 2, где каждая ячейка содержит:
    - id: идентификатор товара
    - P_click: схожесть товара с запросом от 0 до 1 (TF-IDF)
    - P_buy: схожесть товара с запросом от 0 до 1 (sentence-transformers или другая модель)
    - P_look1: схожесть с товаром сверху по изображениям от 0 до 1
    - P_look2: схожесть с товаром слева/справа по изображениям от 0 до 1
    
    Параметры:
    -----------
    query : str
        Текстовый запрос пользователя
    n_products : int
        Количество товаров (должно быть rows * cols)
    vectorizer : TfidfVectorizer
        Векторизатор TF-IDF
    product_vectors : sparse matrix
        TF-IDF векторы товаров
    data_df : pandas.DataFrame
        DataFrame с данными о товарах
    rows : int
        Количество строк в матрице
    cols : int
        Количество колонок в матрице (обычно 2)
    embedding_model : optional
        Модель для эмбеддингов (sentence-transformers) для P_buy
    image_model : optional
        Модель VGG16 для извлечения эмбеддингов изображений
    w2v_model : optional
        Уже загруженная модель Word2Vec (если есть)
    
    Возвращает:
    -----------
    matrix : list of list of dict
        Матрица размера rows * cols, где каждый элемент - словарь с ключами:
        - 'id': str
        - 'P_click': float (0-1)
        - 'P_buy': float (0-1)
        - 'P_look1': float (0-1) или None если нет товара сверху
        - 'P_look2': float (0-1) или None если нет товара слева/справа
    """
    # Генерируем выдачу товаров
    results = generate_feed_tfidf(query, n_products, vectorizer, product_vectors, data_df, rows=rows)
    
    # Вычисляем P_click для каждого товара (TF-IDF схожесть с запросом)
    query_processed = preprocess_query(query)
    query_vector = vectorizer.transform([query_processed])
    
    # Получаем индексы товаров в исходном DataFrame
    product_indices = [item['vector_index'] for item in results]
    product_vectors_subset = product_vectors[product_indices]
    
    tfidf_similarities = cosine_similarity(query_vector, product_vectors_subset)[0]
    p_click_values = (tfidf_similarities + 1) / 2
    p_click_values = np.sqrt(p_click_values)
    p_click_values = np.clip(p_click_values - 0.2, 0, 1)
    
    # Вычисляем P_buy для каждого товара (sentence-transformers схожесть)
    if embedding_model is None:
        embedding_model = get_embedding_model(w2v_model=w2v_model)
    
    p_buy_values = []
    for item in results:
        prob, _ = calculate_purchase_probability(query, item['product_name'], embedding_model=embedding_model, data_df=data_df, w2v_model=w2v_model)
        p_buy_values.append(max(0, prob - 0.2))
    
    # Вычисляем эмбеддинги изображений для всех товаров
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.utils import get_file
    
    if image_model is None:
        image_model = VGG16(weights='imagenet', include_top=False)
    
    def extract_image_embedding(img_url, model):
        """Извлекает эмбеддинг изображения"""
        try:
            if isinstance(img_url, str) and '|' in img_url:
                img_url = img_url.split('|')[0]
            if not img_url or pd.isna(img_url):
                return np.zeros((512,))
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                img_path = get_file(temp_path, origin=img_url)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                features = model.predict(img_array, verbose=0)
                embedding = features.flatten()
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if np.all(embedding == 0):
                    return np.random.rand(512) * 0.01
                return embedding
            except:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return np.random.rand(512) * 0.01
        except:
            return np.random.rand(512) * 0.01
    
    # Извлекаем эмбеддинги для всех товаров
    image_embeddings = []
    for item in results:
        emb = extract_image_embedding(item['image_url'], image_model)
        image_embeddings.append(emb)
    
    image_embeddings = np.array(image_embeddings)
    
    image_similarity_matrix = cosine_similarity(image_embeddings)
    image_similarity_normalized = (image_similarity_matrix + 1) / 2
    image_similarity_normalized = np.clip(image_similarity_normalized, 0, 1)
    
    # Формируем матрицу
    matrix = []
    for row in range(rows):
        matrix_row = []
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(results):
                matrix_row.append(None)
                continue
            
            item = results[idx]
            
            p_look1 = 0.0
            if row > 0:
                up_idx = (row - 1) * cols + col
                if up_idx < len(results):
                    p_look1 = float(image_similarity_normalized[idx, up_idx])
            
            p_look2 = 0.0
            if col > 0:
                left_idx = row * cols + (col - 1)
                if left_idx < len(results):
                    p_look2 = float(image_similarity_normalized[idx, left_idx])
            elif col < cols - 1:
                right_idx = row * cols + (col + 1)
                if right_idx < len(results):
                    p_look2 = float(image_similarity_normalized[idx, right_idx])
            
            matrix_row.append({
                'id': item['id'],
                'P_click': float(p_click_values[idx]),
                'P_buy': float(p_buy_values[idx]),
                'P_look1': float(p_look1),
                'P_look2': float(p_look2)
            })
        matrix.append(matrix_row)
    
    return matrix


def display_feed_matrix(matrix, rows=2, cols=2):
    """
    Отображает матрицу выдачи в удобном виде
    
    Параметры:
    -----------
    matrix : list of list of dict
        Матрица от generate_feed_matrix
    rows : int
        Количество строк
    cols : int
        Количество колонок
    """
    print("=" * 100)
    print("МАТРИЦА ВЫДАЧИ ТОВАРОВ")
    print("=" * 100)
    
    for row in range(rows):
        print(f"\nСтрока {row + 1}:")
        print("-" * 100)
        for col in range(cols):
            if matrix[row][col] is None:
                print(f"  Колонка {col + 1}: [пусто]")
                continue
            
            item = matrix[row][col]
            print(f"\n  Колонка {col + 1}:")
            print(f"    ID: {item['id']}")
            print(f"    P_click (TF-IDF): {item['P_click']:.4f}")
            print(f"    P_buy (sentence-transformers): {item['P_buy']:.4f}")
            print(f"    P_look1 (сверху): {item['P_look1']:.4f}")
            print(f"    P_look2 (слева/справа): {item['P_look2']:.4f}")
        
        if row < rows - 1:
            print("\n" + "=" * 100)
    
    print("\n" + "=" * 100)
    
    # Сводная таблица
    flat_data = []
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] is not None:
                item = matrix[row][col].copy()
                item['row'] = row + 1
                item['col'] = col + 1
                flat_data.append(item)
    
    if flat_data:
        df = pd.DataFrame(flat_data)
        print("\nСводная таблица:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    # Загружаем данные и модели
    print("Загрузка данных и моделей...")
    data, tifd, tf, w2v_model = load_all()
    
    # Параметры запроса
    query = "black T-shirt"
    n_products = 16
    rows = 8
    cols = 2

    # Генерируем матрицу
    matrix = generate_feed_matrix(
        query=query,
        n_products=n_products,
        vectorizer=tifd,
        product_vectors=tf,
        data_df=data,
        rows=rows,
        cols=cols,
        w2v_model=w2v_model
    )

    # Отображаем результаты
    display_feed_matrix(matrix, rows=rows, cols=cols)
