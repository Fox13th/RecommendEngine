import json
import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
import redis
from dotenv import load_dotenv

from surprise import SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

load_dotenv()


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
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> list:
        """Возвращает список рекомендованных фильмов для заданного пользователя."""
        pass

    def get_all_recommendations(self):
        """Возвращает список рекомендованных фильмов для всех пользователей."""
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

    def get_user_recommendations(self, user_id: str, n_recommendations: int = 5) -> list:
        """Возвращает список рекомендованных фильмов для заданного пользователя."""

        all_movie_ids = self.ratings_df['movie_id'].unique()
        rated_movies = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].values
        movies_to_predict = [movie for movie in all_movie_ids if movie not in rated_movies]

        predictions = [self.model.predict(user_id, movie) for movie in movies_to_predict]
        predictions.sort(key=lambda x: x.est, reverse=True)

        top_n_recommendations = predictions[:n_recommendations]
        return [(pred.iid, pred.est) for pred in top_n_recommendations]

    def get_all_recommendations(self):
        """Возвращает список рекомендованных фильмов для всех пользователей."""

        all_users = self.ratings_df['user_id'].unique()
        for user in all_users:
            user_recommend = self.get_user_recommendations(user, n_recommendations=10)
            logger.info(f'Рекомендация для пользователя {user}: {user_recommend}')
            redis_db.set(user, json.dumps(user_recommend))
            redis_db.expire(user, int(os.getenv('REDIS_EXPIRATION')))

            # Пример получения списка рекомендаций
            # get_list = json.loads(redis_db.get(user))
            # logger.info(get_list)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    redis_db = redis.StrictRedis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), db=0)

    recomend_data = CollaborativeFilter('matrix_surprise.csv')
    recomend_data.train_model()
    recomend_data.get_all_recommendations()
