import logging
from abc import ABC, abstractmethod

import pandas as pd

from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


class RecommendService(ABC):

    def __init__(self, data_file: str):
        """Инициализирует сервис рекомендаций с данными из файла."""
        self.model = SVD()
        self.ratings_df = pd.read_csv(data_file)

    @abstractmethod
    def train_model(self) -> list:
        """Обучает модель на основе данных рейтингов и возвращает предсказания."""
        pass

    @abstractmethod
    def get_recommendations(self, user_id: int, n_recommendations: int = 5) -> list:
        """Возвращает список рекомендованных фильмов для заданного пользователя."""
        pass


class CollaborativeFilter(RecommendService):

    def train_model(self) -> list:
        """Обучает модель на основе данных рейтингов и возвращает предсказания."""
        reader = Reader(rating_scale=(1, 5))
        input_data = Dataset.load_from_df(self.ratings_df[['user_id', 'movie_id', 'rating']], reader)

        trainset, testset = train_test_split(input_data, test_size=0.25)

        self.model.fit(trainset)

        predictions = self.model.test(testset)
        accuracy.rmse(predictions)

        return predictions

    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> list:
        """Возвращает список рекомендованных фильмов для заданного пользователя."""

        all_movie_ids = self.ratings_df['movie_id'].unique()
        rated_movies = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values
        movies_to_predict = [movie for movie in all_movie_ids if movie not in rated_movies]

        predictions = [self.model.predict(user_id, movie) for movie in movies_to_predict]
        predictions.sort(key=lambda x: x.est, reverse=True)

        top_n_recommendations = predictions[:n_recommendations]
        return [(pred.iid, pred.est) for pred in top_n_recommendations]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    user_id = '27a06871-401f-4c17-b8d7-56540dc755cb'

    recomend_data = CollaborativeFilter('matrix_surprise.csv')
    recomend_data.train_model()
    recommendations = recomend_data.get_recommendations(user_id, n_recommendations=3)

    logger.info(f'Рекомендации для пользователя: {recommendations}')
