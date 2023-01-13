# :movie_camera: :clapper: Model Based CF for Movies Recommendation System using SVD

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

One way Netflix overcomes these challenges is by using machine learning. Machine learning algorithms such as collaborative filtering and content-based 
filtering can be used to analyze a user's viewing history and preferences, and to make personalized recommendations based on that data.

## Different types of Content Filtering
Content-Based Filtering: This is a recommendation system that suggests items to users based on their previous interactions with items that have similar characteristics. For example, if a user has previously rated movies of a certain genre highly, a content-based filtering system would recommend other movies of that genre to the user.

Memory-Based Collaborative Filtering: This is a recommendation system that suggests items to users based on the interactions of similar users. For example, if a user has similar movie ratings to another user, a memory-based collaborative filtering system would recommend the same movies to both users.

Model-Based Collaborative Filtering: This is a recommendation system that suggests items to users based on a model that has been trained on past interactions. For example, a model-based collaborative filtering system could use matrix factorization to create latent representations of users and items, and then use those representations to make recommendations.

Deep Learning / Neural Network: This is a recommendation system that uses deep learning techniques such as neural networks to make recommendations. These systems can be trained on large amounts of data and can take into account a wide range of features such as user behavior, item metadata, and context.

It is important to note that each of these methods have their own set of advantages and disadvantages and the choice of which method to use will depend on the specific use case and requirements.



## Challenges with Memory Based CF
In previous analysis, I implemeted CF using pearson correlation similarity as 
shown in this 
[notebook.](https://github.com/iqrabismii/MoviesRecommendationSystem/blob/main/CFBasedRecommendation/CollaborativeFiltering_RecommendationSystem.ipynb)

However, there are some drawbacks:
* It does not provide a solution for the issue of new users or items entering the system, known as the cold-start problem.
* It struggles with sparse data, making it difficult to find similar users that have rated the same items.
* It becomes less effective when new users or items without any ratings enter the system.
* It tends to recommend popular items, rather than providing personalized recommendations.
* It doesn’t scale particularly well to massive datasets, especially for real-time recommendations based on user behavior similarities — which takes a lot of computations

Hence to overcome this issue, we need to apply dimensionality reduction technique which can be implemented using __Matrix Factorisation__


## Model Based Learning

__Model-based Collaborative Filtering__ is a technique that utilizes matrix factorization to make recommendations. 
It is considered a more advanced approach than Memory-based Collaborative Filtering and 
is better suited for handling large and __sparse datasets__. 
The goal of MF is to learn the underlying preferences of users and the attributes of 
items from known ratings and use this information to predict unknown ratings. This is achieved by 
__decomposing__ the user-item matrix into a low-rank structure, where the rows contain latent vectors.
By multiplying these low-rank matrices together, we can approximate the original matrix and 
fill in the missing entries. This approach allows for better scalability and handling of sparsity in comparison to __memory-based methods__.

## Support Vector Decomposition(SVD)
SVD is an algorithm that decomposes a matrix into a lower rank approximation of the original matrix. 
It does this by breaking down the matrix into three parts: two unitary matrices and a diagonal matrix.
The input data matrix is represented by A, the left singular vectors is represented by U, the diagonal matrix 
of singular values is represented by Sigma, and the right singular vectors is represented by V. U and V are column orthonormal 
and they represent different things. U represents how much users "like" each feature, and V represents how relevant each feature is to each movie. 
To get the lower rank approximation, we keep only the top features, which can be thought of as the underlying tastes and preferences vectors.
![Screenshot 2023-01-13 at 1 22 03 AM](https://user-images.githubusercontent.com/108056063/212284374-adcc89eb-4166-4f22-afad-147d78c11741.png)



## Decomposing the Matrix

The returned values are the three matrices of the SVD decomposition:

* U: the left singular vectors of the input matrix Ratings_demeaned, It represents how much users "like" each feature.
* sigma: the diagonal matrix of singular values, It is a 1-D array containing the singular values in descending order. These values are essentially weights/strengths of each concept.
* Vt: the transpose of the right singular vectors of the input matrix Ratings_demeaned, It represents how relevant each feature is to each movie.
Details are provided in this [notebook.](https://github.com/iqrabismii/MoviesRecommendationSystem/blob/main/CFBasedRecommendation/ModelBasedCF_RecommendationSystem.ipynb)

## Making Predictions from the Decomposed Matrices

I computed movie ratings predictions for every user by following the 
mathematical process and multiplying matrices U, sigma, and Vt to get the 
lower rank approximation of the original matrix. However, before doing so, 
I added the mean ratings for each user back into the predictions in order to obtain the actual star ratings predictions
I wrote a function that returns the movies with the highest predicted rating that 
the specified user has not yet rated. To provide more context to the recommendations,
I also merged in movie content information such as genre and title even though 
I didn't use any explicit movie content features in the prediction process. More details are provided in the [notebook.](https://github.com/iqrabismii/MoviesRecommendationSystem/blob/main/CFBasedRecommendation/ModelBasedCF_RecommendationSystem.ipynb)

