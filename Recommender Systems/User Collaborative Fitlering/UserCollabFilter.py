import sys
import math

""" 
Template adapted from:
https://github.com/ZwEin27/User-based-Collaborative-Filtering/blob/master/collabFilter.py.
Adjustments made: 
    1. Normalizing the ratings of neighbors when predicting a rating for a new (user, item) pair
    2. Use the k nearest neighbors to our users, who have the item in common (instead of just the absolute k nearest) 

"""
class collaborative_filtering:
    def __init__(self, file_name, user_id, movie, n):
        self.file_name = file_name
        self.user_id = user_id
        self.movie = movie
        self.n = n
        self.dataset = None
        self.uu_dataset = None
        self.ii_dataset = None

    def initialize(self):
        """ 
        Initialize parameters using the data loader
        
        """

        # load data
        self.dataset, self.uu_dataset= self.data_loader(self.file_name)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """ Calculate usesr similarity, here, using Pearson Correlation"""
    """ Another option would be to use cosine similarity           """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def user_average_rating(self, user):
        user_average = 0.0

        user_data = self.uu_dataset[user]
        total_ratings = len(user_data)


        for (movie, rating) in user_data.items():
            user_average += rating
        user_average = user_average / total_ratings
        
        return user_average


    def common_items(self, userx, usery):
        userx_data = self.uu_dataset[userx]
        usery_data = self.uu_dataset[usery]

        intersection = userx_data.keys() & usery_data.keys()
        
        return intersection



    def pearson_correlation(self, userx, usery):

        ## Get user data:
        userx_data = self.uu_dataset[userx]
        usery_data = self.uu_dataset[usery]

        ## Get user average ratings:
        userx_average = self.user_average_rating(userx)
        usery_average = self.user_average_rating(usery)

        ## Get common items between both users:
        items_xy = self.common_items(userx, usery)
        #print(items_xy)

        num = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for item in items_xy: 
            x_rating = userx_data[item]
            y_rating = usery_data[item]

            num += (x_rating - userx_average)*(y_rating - usery_average)
            denom_x += (x_rating - userx_average)**2
            denom_y += (y_rating - usery_average)**2

        denom_x = math.sqrt(denom_x)
        denom_y = math.sqrt(denom_y)

        sim_xy = num/(denom_x * denom_y)

        return sim_xy


    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """                    K-Nearest Neighbors                     """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def k_nearest_neighbors(self, user, item, n):
        neighbors = {}
        valid_neighbors = {}
        for neighbor in self.uu_dataset.keys():
            if neighbor == user:
                continue 
            upc = self.pearson_correlation(user, neighbor)
            neighbors[neighbor] = upc

            ## Choose the neighbors which have seen the item/movie:
            if item in self.uu_dataset[neighbor].keys():
                valid_neighbors[neighbor] = upc

        sorted_neighbors = sorted(neighbors.items(), key=lambda x:x[1], reverse = True)
        sorted_valid_neighbors = sorted(valid_neighbors.items(), key=lambda x:x[1], reverse = True)

        return sorted_valid_neighbors[:n]



    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """                          Predict                           """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def predict(self, user, item, k_nearest_neighbors):

        num = 0.0
        denom = 0.0

        average_rating = self.user_average_rating(user)

        for neighbor in k_nearest_neighbors: 
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]
            neighbor_rating = self.uu_dataset[neighbor_id][item]
            neighbor_average_rating = self.user_average_rating(neighbor_id)

            # Adjust/normalize ratings by subtracting neighbor's average and later, adding user average rating
            ## Raters rate on different scales, and this accounts for this bias:
            num += neighbor_similarity * (neighbor_rating - neighbor_average_rating)
            # num += neighbor_similarity * neighbor_rating 
            denom += neighbor_similarity
        result = average_rating + num/denom
        #result = num/denom
        return result 






    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """                    Helper Functions                        """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def data_loader(self, file_name):
        """ 
        Load dataset and return three separate files dataset, uu_dataset, ii_dateset
        
        """

        input_file = open(file_name)
        dataset = []
        uu_dataset = {}
        # ii_dataset = {}

        for line in input_file:
            row = str(line).split("\t")
            row[2] = row[2].strip()
            row[1] = float(row[1])

            dataset.append(row)

        
            uu_dataset.setdefault(row[0], {})
            uu_dataset[row[0]].setdefault(row[2], float(row[1]))
            # uu_dataset[row[0]].append([row[2],row[1]])

        return dataset, uu_dataset
        



# test 
file_name = "ratings-dataset.tsv"
user_id =  "Kluver" ##"Flesh"   ### "Kluver"
movie = "The Fugitive" ## 'Pulp Fiction' ### "The Fugitive"
n = 10

cf = collaborative_filtering(file_name, user_id, movie, n)
cf.initialize()

print(cf.k_nearest_neighbors(user_id, movie, n))

print(cf.uu_dataset['6d87994b-d015-485b-926c-b2ddead6efd7']['Pulp Fiction'])
print(cf.uu_dataset['14150340-dc4b-4eca-a845-d92ac49281ea']['Pulp Fiction'])
print(cf.uu_dataset['8ccfa5d6-6f0b-407f-a463-0e1f745f9dad']['Pulp Fiction'])
print(cf.uu_dataset['vikram']['Pulp Fiction'])


# data = cf.data_loader(file_name)
#print(data[1])



# user_x = '6632e5b3-89cc-461c-aaf7-06e69e634333'
# user_y = 'k279'


#cf.pearson_correlation(user_x, user_y)
# print(cf.pearson_correlation("Flesh", "Nathan_Studanski"))
#print(cf.common_items("Flesh", "Nathan_Studanski"))
#print("\n")
#print(cf.uu_dataset["Nathan_Studanski"]["Beauty and the Beast"])

# print(cf.uu_dataset["Nathan_Studanski"])
# print(cf.uu_dataset["Flesh"])


# userx_data = cf.uu_dataset["Nathan_Studanski"]
# usery_data = cf.uu_dataset["Flesh"]

# intersection = userx_data.keys() & userx_data.keys()
# print(intersection)
        

# # Test : 
# my_dict = {"Farida": 1.5, "eem": 4.5, "suzy":2.5}

# list = sorted(my_dict.items(), key=lambda x:x[1], reverse = True)

# print(list[:2])

k_nearest_neighbors = cf.k_nearest_neighbors(user_id, movie, n)
 

prediction = cf.predict(user_id, movie, k_nearest_neighbors)
print(prediction)

#print(cf.uu_dataset['Flesh']['Pulp Fiction'])
#print("\n")
#print(cf.uu_dataset['Nathan_Studanski']['Pulp Fiction'])