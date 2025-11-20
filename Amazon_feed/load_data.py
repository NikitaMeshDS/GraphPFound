import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

try:
    import scipy.linalg
    if not hasattr(scipy.linalg, 'triu'):
        from numpy import triu
        scipy.linalg.triu = triu
except:
    pass


def load_amazon_data(data_path=None):
    """Загружает данные Amazon Fashion из JSON файла"""
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
        
        possible_paths = [
            os.path.join(base_dir, filename),
            filename,
            "/Users/nikitamesh/GraphPFound/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Не найден файл данных.")
    
    print(f"Загрузка данных из {data_path}...")
    data = pd.read_json(data_path, lines=True)
    print(f"Загружено {len(data)} записей")
    return data


def prepare_tfidf(data, ngram_range=(1, 2), min_df=10):
    """Создает TF-IDF векторизатор и векторы для названий товаров"""
    print("Создание TF-IDF векторизатора...")
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    vectors = vectorizer.fit_transform(data['product_name'])
    print(f"TF-IDF векторизатор создан. Размерность: {vectors.shape}")
    return vectorizer, vectors


def load_word2vec_model(w2v_path=None):
    """Загружает предобученную модель Word2Vec через gensim.downloader"""
    try:
        import gensim.downloader as api
        
        models = [
            "glove-wiki-gigaword-100",
            "fasttext-wiki-news-subwords-300",
            "word2vec-google-news-300"
        ]
        
        for model_name in models:
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
        print(f"Ошибка загрузки Word2Vec: {e}")
        return None


def load_all(data_path=None, w2v_path=None, ngram_range=(1, 2), min_df=10):
    """Загружает все необходимые данные и модели"""
    data = load_amazon_data(data_path)
    vectorizer, vectors = prepare_tfidf(data, ngram_range=ngram_range, min_df=min_df)
    w2v_model = load_word2vec_model(w2v_path)
    return data, vectorizer, vectors, w2v_model


if __name__ == "__main__":
    data, tifd, tf, w2v_model = load_all()
    print(f"  - Количество товаров: {len(data)}")
    print(f"  - Размерность TF-IDF: {tf.shape}")
    print(f"  - Word2Vec модель: {'Загружена' if w2v_model is not None else 'Не загружена'}")
