"""
Модуль для загрузки и предобработки данных Amazon Fashion
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Патч для исправления проблемы совместимости scipy/gensim
# Применяем до импорта gensim
try:
    import scipy.linalg
    if not hasattr(scipy.linalg, 'triu'):
        try:
            from numpy import triu
            scipy.linalg.triu = triu
        except:
            pass
except:
    pass


def load_amazon_data(data_path=None):
    """
    Загружает данные Amazon Fashion из JSON файла
    
    Параметры:
    -----------
    data_path : str, optional
        Путь к файлу данных. Если None, пытается найти стандартный путь.
    
    Возвращает:
    -----------
    data : pandas.DataFrame
        DataFrame с данными о товарах
    """
    if data_path is None:
        # Определяем базовую директорию (где находится этот файл)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
        
        # Пробуем стандартные пути
        possible_paths = [
            os.path.join(base_dir, filename),  # В той же директории, что и скрипт
            filename,  # В текущей рабочей директории
            "/Users/nikitamesh/GraphPFound/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
            "/kaggle/input/amazon-fashion-products-2020/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
            f"data/{filename}"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Не найден файл данных. Укажите путь к файлу в параметре data_path")
    
    print(f"Загрузка данных из {data_path}...")
    data = pd.read_json(data_path, lines=True)
    print(f"Загружено {len(data)} записей")
    return data


def prepare_tfidf(data, ngram_range=(1, 2), min_df=10):
    """
    Создает TF-IDF векторизатор и векторы для названий товаров
    
    Параметры:
    -----------
    data : pandas.DataFrame
        DataFrame с данными о товарах (должен содержать колонку 'product_name')
    ngram_range : tuple
        Диапазон n-грамм для TF-IDF (по умолчанию (1, 2))
    min_df : int
        Минимальная частота документа для включения слова в словарь
    
    Возвращает:
    -----------
    vectorizer : TfidfVectorizer
        Обученный векторизатор TF-IDF
    vectors : sparse matrix
        TF-IDF векторы всех товаров
    """
    print("Создание TF-IDF векторизатора...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    vectors = vectorizer.fit_transform(data['product_name'])
    print(f"TF-IDF векторизатор создан. Размерность: {vectors.shape}")
    return vectorizer, vectors


def load_word2vec_model(w2v_path=None):
    """
    Загружает предобученную модель Word2Vec
    Если файл не найден, пытается загрузить через gensim.downloader
    
    Параметры:
    -----------
    w2v_path : str, optional
        Путь к файлу модели Word2Vec. Если None, пытается найти стандартный путь или загрузить через gensim.downloader.
    
    Возвращает:
    -----------
    w2v_model : KeyedVectors или None
        Загруженная модель Word2Vec или None, если не найдена
    """
    try:
        from gensim.models import KeyedVectors
        
        # Сначала пробуем загрузить из файла, если путь указан
        if w2v_path is not None and os.path.exists(w2v_path):
            print(f"Загрузка модели Word2Vec из {w2v_path}...")
            w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            print("Модель Word2Vec загружена из файла")
            return w2v_model
        
        # Если путь не указан или файл не найден, пробуем стандартные пути
        if w2v_path is None:
            possible_paths = [
                '/kaggle/input/gensimmodel/GoogleNews-vectors-negative300.bin',
                'GoogleNews-vectors-negative300.bin',
                os.path.join(os.path.expanduser('~'), 'GoogleNews-vectors-negative300.bin')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"Загрузка модели Word2Vec из {path}...")
                    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)
                    print("Модель Word2Vec загружена")
                    return w2v_model
        
        # Если файл не найден, пробуем загрузить более легкую модель через gensim.downloader
        print("Файл Word2Vec не найден. Пытаемся загрузить легкую модель через gensim.downloader...")
        try:
            import gensim.downloader as api
            
            # Пробуем более легкие модели в порядке приоритета
            light_models = [
                "glove-wiki-gigaword-100",  # ~128MB, 100 размерность
                "fasttext-wiki-news-subwords-300",  # ~958MB, но быстрее загружается
                "word2vec-google-news-300"  # ~1.6GB, только если другие не работают
            ]
            
            for model_name in light_models:
                try:
                    print(f"Попытка загрузки модели '{model_name}'...")
                    w2v_model = api.load(model_name)
                    print(f"Модель '{model_name}' загружена успешно")
                    return w2v_model
                except Exception as e:
                    print(f"Не удалось загрузить '{model_name}': {e}")
                    continue
            
            print("Не удалось загрузить ни одну модель")
            return None
        except Exception as e:
            print(f"Ошибка при работе с gensim.downloader: {e}")
            print("Будет использован TF-IDF fallback для P_buy")
            return None
            
    except ImportError as e:
        if 'triu' in str(e) or 'gensim' in str(e):
            print(f"Проблема совместимости scipy/gensim: {e}")
            print("Попробуйте обновить gensim: pip install --upgrade gensim")
            print("Или понизить версию scipy: pip install 'scipy<1.11'")
            print("Будет использован TF-IDF fallback для P_buy")
        else:
            print(f"Ошибка импорта: {e}")
        return None
    except Exception as e:
        print(f"Ошибка загрузки Word2Vec: {e}")
        return None


def load_all(data_path=None, w2v_path=None, ngram_range=(1, 2), min_df=10):
    """
    Загружает все необходимые данные и модели
    
    Параметры:
    -----------
    data_path : str, optional
        Путь к файлу данных Amazon Fashion
    w2v_path : str, optional
        Путь к файлу модели Word2Vec
    ngram_range : tuple
        Диапазон n-грамм для TF-IDF
    min_df : int
        Минимальная частота документа для TF-IDF
    
    Возвращает:
    -----------
    data : pandas.DataFrame
        DataFrame с данными о товарах
    vectorizer : TfidfVectorizer
        Обученный векторизатор TF-IDF
    vectors : sparse matrix
        TF-IDF векторы всех товаров
    w2v_model : KeyedVectors или None
        Загруженная модель Word2Vec или None
    """
    data = load_amazon_data(data_path)
    vectorizer, vectors = prepare_tfidf(data, ngram_range=ngram_range, min_df=min_df)
    w2v_model = load_word2vec_model(w2v_path)
    
    return data, vectorizer, vectors, w2v_model


if __name__ == "__main__":
    # Пример использования
    data, tifd, tf, w2v_model = load_all()
    print(f"\nДанные загружены:")
    print(f"  - Количество товаров: {len(data)}")
    print(f"  - Размерность TF-IDF: {tf.shape}")
    print(f"  - Word2Vec модель: {'Загружена' if w2v_model is not None else 'Не загружена'}")

