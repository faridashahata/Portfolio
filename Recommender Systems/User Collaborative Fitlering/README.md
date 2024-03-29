## Overview
This is an implementation of user-collaborative filtering for the author to practice while learning the subject from a Coursera course from the University of Minnesota: Nearest Neighbor Collaborative Filtering, adapted from the following implementation: https://github.com/ZwEin27/User-based-Collaborative-Filtering/tree/master.

The goal is to predict a movie rating for a new (user, item) pair based on previous user ratings for movies, using the k nearest neighbors to the user based on Pearson Correlation.

## Data
The rating dataset is obtained from the following git link: https://github.com/ZwEin27/User-based-Collaborative-Filtering/blob/master/ratings-dataset.tsv

It consists of 1441 lines, each corresponding to a different rating event of a movie. Each line consists of a user_id a rating out of 5 and the movie title for which the rating applies. Here is a small sample: 




| User ID  | Rating | Movie Title |
| ---------| -------| ------------|
|JosephIsAwesome	|3.5|	Men in Black|
|JosephIsAwesome	|5.0	|Braveheart|
|bbbdc2b5-e28c-4cad-b59d-5043999e84e9	|4.0	|Donnie Darko|
|bbbdc2b5-e28c-4cad-b59d-5043999e84e9	|4.5	|V for Vendetta|



## Code Structure

1. Calculate Pearson Correlation between two users
  
   .<img width="398" alt="Screenshot 2023-06-26 at 5 44 48 PM" src="https://github.com/faridashahata/Portfolio/assets/113303940/bc6f337a-21ca-42e9-a7c5-eb3948d99e53">,
   
   where $I$ is the overlap of items between users $u$ and $v$.

4. Retreive based on these correlation numbers the $k$ nearest neighbors of a user. Retrieve the first $k$ users who have the highest correlation factor with our user.
5. Predict the movie rating based on the user's average ratings, the $k$ nearest neighbors' average ratings and their ratings of the movie itself.
   
   The predicted rating is calculated as follows:

   
   <img width="367" alt="Screenshot 2023-06-26 at 4 57 56 PM" src="https://github.com/faridashahata/Portfolio/assets/113303940/14c78623-1b90-4298-a941-bb4173917581">,

   
   where $V$ denotes the $k$ nearest neighbors of user $u$, who is rating item $i$, and $w_{uv}$ denotes the similarity between users $u$ and $v$. Also, $\bar{r}_u$ is the average rating of user $u$ and $r_{vi}$ is the rating of neighbor $v$ to item $i$. We note that we adjust/normalize ratings by subtracting neighbor's average and later, adding user average rating. This is because raters rate on different scales, and this accounts for this bias.

## References


Ekstrand M., Konstan J., Nearest Neighbor Collaborative Filtering. Coursera. https://www.coursera.org/lecture/collaborative-filtering/user-user-collaborative-filtering-vxEBZ.

https://github.com/ZwEin27/User-based-Collaborative-Filtering/tree/master

