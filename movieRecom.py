import os
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

def init_spark_context():
	conf = SparkConf().setAppName("MovieRecommendationSystem")
	sc = SparkContext(conf=conf)
	return sc

sc  = init_spark_context()
dataset_path = os.path.join('datasets','ml-latest')

ratings_path = os.path.join(dataset_path,'ratings.csv')
ratings_RDD_unflitered = sc.textFile(ratings_path)
ratings_header = ratings_RDD_unflitered.take(1)[0]

# pyspark RDD for ratings
ratings_RDD = ratings_RDD_unflitered.filter(lambda line: line!=ratings_header).map(lambda line: line.split(",")).map(lambda tokens: int(tokens[0],int(tokens[1]),float(tokens[2]))).cache()


movies_path = os.path.join(dataset_path,'movies.csv')
movies_RDD_unfiltered = sc.textFile(movies_path)
movies_header = movies_RDD_unfiltered.take(1)[0]

# pyspark RDD for movies and titles
movies_RDD = movies_RDD_unfiltered.filter(lambda line: line!=movies_header).map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
movies_titles_RDD = movies_RDD.map(lambda x: (int(x[0]),x[1])).cache()

# Pre-calculate movies ratings counts
movie_ID_with_ratings_RDD = ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# Recommendation system using Alternating Least Squares
rank = 8
seed = 5L
iterations = 10
regularization_parameter = 0.1
model = ALS.train(ratings_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)

for_movies_RDD = sc.parallelize(movie_ids).map(lambda x: (user_id,x))

predicted_RDD = model.predictAll(for_movies_RDD)
predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
# All predicted values of rating, title and count put together in an RDD
predicted_RDD = predicted_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
predicted_RDD = predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
MSE = predicted_RDD.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

ratings = predicted_RDD.collect()
print(ratings)





