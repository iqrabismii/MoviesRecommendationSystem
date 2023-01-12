# :movie_camera: :clapper: Content and Collaborative Based Movies Recommendation System

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)


## Introduction 
__Netflix__ is a streaming platform that offers a wide variety of movies and TV shows to its subscribers. 
One of the key features of Netflix is its recommendation system, which suggests content to users 
based on their viewing history and preferences. The recommendation system is designed to help users discover new movies and 
TV shows that they may be interested in watching, and to make the overall streaming experience more enjoyable and personalized.

The Netflix recommendation system is based on a combination of 
__Machine Learning__ and __Data Mining Techniques__. It uses a user's 
viewing history, search history, and ratings to make personalized recommendations. 
The system also takes into account the popularity and the release date of the movie
and the genre of the movie. Additionally, it uses __Collaborative Filtering__ and __Content-based 
Filtering__ to make recommendations. Collaborative filtering makes recommendations based on the 
behavior of similar users, while content-based filtering makes recommendations based on the characteristics of the movie.

The __Netflix Recommendation System__ is constantly being updated and improved to provide 
better and more relevant recommendations to users. It is one of the key features that sets Netflix
apart from other streaming platforms and has played a significant role in the company's success. 

 
## Challenges

There are several challenges that Netflix faces when 
recommending movies to its users. One major challenge is the vast 
amount of content available on the platform. With thousands of movies 
and TV shows to choose from, it can be difficult to make personalized 
recommendations that are relevant to each individual user.

Another challenge is the diversity of user preferences. Different users 
have different tastes and preferences when it comes to movies and TV shows, 
which can make it difficult to make recommendations that will appeal to everyone.
Additionally, users can change their preferences over time, making it difficult for the recommendation system to keep up.

One way Netflix overcomes these challenges is by using machine learning and cosine 
similarity. Machine learning algorithms such as collaborative filtering and content-based 
filtering can be used to analyze a user's viewing history and preferences, and to make personalized recommendations based on that data.

__Cosine similarity__ is a measure of similarity between two non-zero vectors of an inner 
product space. It can be used to find the similarity between the items, for example, 
the similarity between two movies. By using cosine similarity, the system can identify movies
that are similar in terms of genre, plot, or other characteristics, and recommend those movies 
to users who have shown an interest in similar movies. This can help to increase the relevance and accuracy of recommendations for users.


Overall, the use of __Machine Learning Algorithms__ and __Cosine Similarity__ allows Netflix to 
overcome the challenges of personalizing recommendations for its diverse user base, and 
to provide a more enjoyable and personalized streaming experience.




 


## Data Science and Machine Learning 

With the help of __machine learning__, 
content and collaborative filtering can be applied. The data set used to built recommendation system was from
[Kaggle.](https://github.com/iqrabismii/MoviestRecommendationSystem/blob/main/MoviesRecommendationSystem/tmdb_5000_credits.csv)

## Exploratory Data Analysis (EDA)

Visualised the distribution of average rating and number of rating for all movies using bar plot and histogram. EDA can be found here in this 
[notebook.](https://github.com/iqrabismii/MoviestRecommendationSystem/blob/main/MoviesRecommendationSystem/MoviesRecommendationSystem.ipynb)
we observed that average vote across all genres are same but vote count is different. Hence, if select top movies by 
rating then the results might be biased. For eg, if average rating rated by 10 people is  9 whereas  average rating rated by  
100 people is 8.9 then movies rated by more users is much better. 
However, algorithm will be biased towards higher rated value. Hence to overcome this issue we will use weighted rating.



## Simple Recommendation System to suggest top 25 movies based of Weighted Rating
Weighted Rating can be calculated by taking vote count, vote rating, min number of votes required, mean rating value. This helped to suggest top high performing
movies. Details can be found [here.](https://github.com/iqrabismii/MoviestRecommendationSystem/blob/main/MoviesRecommendationSystem/MoviesRecommendationSystem.ipynb)

## Content Based Filtering,  based on features of movies such as movie description, directors, actors, title and genre

For __Content Based Filtering__,  built a recommendation system that suggests movies that are similar to a specific movie. we will 
calculate the cosine similarity between all movies based on
their plot descriptions, and recommend movies that have a similarity score above a certain threshold.

To compute the similarity, we will find the word vectors using TF-IDF. 
TF-IDF score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs.
To compute the similarity,we will calculate cosine similarity between vectors. we will use linear_kernel() as it is faster.
The output will be a matrix of 4830x4830 shape, which means the cosine similarity score between each movie's overview 
and every other movie's overview. To map these similarity score with movies, we need to do reverse mapping.
To create a recommendation function, the following steps will be taken: 
1. Obtain the index of a movie using its title.
2. Compute a list of cosine similarity scores between the selected movie and all other movies, and convert it into a list of tuples where the first element is the movie's index and the second is the similarity score.
3. Sort the list of tuples based on similarity scores.
4. Retrieve the top 10 elements from the sorted list, excluding the first element which represents the selected movie.
5. Return the titles of the movies corresponding to the indices of the top 10 elements.
More details about Content Based Filtering can be found [here.](https://github.com/iqrabismii/MoviestRecommendationSystem/blob/main/MoviesRecommendationSystem/MoviesRecommendationSystem.ipynb)

## Collaborative Based Filtering based on  similar users

__Collaborative filtering__ is a technique used in recommendation systems that 
suggests items to users based on the behavior of similar users. In the case of Netflix,
it suggests movies to users based on the movies that similar users have watched and liked. 
The following are the general steps used in a collaborative filtering system for movies based on correlation score:

Data Collection: The first step is to collect data on the movies watched by users on Netflix, 
including the movie title and the user's rating for each movie.

Data Preprocessing: The collected data is then preprocessed to remove any irrelevant information 
and to prepare it for the next step. This includes tasks such as dealing with missing values and removing outliers.

User-Item Matrix: The preprocessed data is then used to create a user-item matrix,
which is a table that shows the ratings given by each user to each movie. The entries 
in the matrix are numerical values (indicating the user's rating for the movie on a scale of 1 to 5, for example).

Correlation Calculation: The user-item matrix is then used to 
calculate the correlation score between pairs of users. The correlation 
score is a measure of the similarity between the ratings given by two users, 
and can be calculated using a method such as Pearson correlation coefficient.

Similarity Matrix: The correlation scores are then used to create a similarity matrix, 
which shows the similarity between each pair of users. 
The entries in the matrix are numerical values (indicating the degree of similarity between the users).

Prediction: Once the similarity matrix is created, 
it can be used to make predictions about the ratings 
that a user would give to a movie that they have not yet watched.
This is done by finding the most similar users and using their ratings for the movie to make a prediction.

Recommendation: Once the predictions are made, 
the system can recommend movies to the user that they are
likely to enjoy based on the predicted ratings. This can be done by recommending the movies with the highest predicted ratings.

More details about collaborative-based filtering can be found [here.](https://github.com/iqrabismii/MoviestRecommendationSystem/blob/main/MoviesRecommendationSystem/MoviesRecommendationSystem.ipynb)




 


