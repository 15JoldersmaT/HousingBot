#Housing bot
#using linear regression algorithm to predict housing price based on various factors
#using california  census data from 1990
#warning is from removing langitude and longitude 
#They they can be added back if desired, but you will
#need to add both to your test case


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing 
import matplotlib.pyplot as plt


housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data = data.drop(['Latitude', 'Longitude'], axis=1)
x = data
y = housing.target



def train(x, y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x,y)
    return model


#train the model 
model = train(x,y)

#Used to test model/use
#values repersent (in order) Median income, Median age, avg num of rooms per house,
#avg numb of bedrooms per house
#avg numb of house members

#On data used
#details here: https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
def test():
    inc = float(input('Enter median income(tens of thousands) for block: '))
    age = float(input('Enter median age of house on block : '))
    aveRooms = float(input('Enter avg num of rooms per house : '))
    bPop = float(input('Enter population of block : '))
    avgBRooms = float(input('Enter avg num of bedrooms per house : '))
    avgHMembers = float(input('Enter avg num of house members in block :'))


    x_new = [[inc, age, aveRooms, avgBRooms,bPop,  avgHMembers]]
    y_new = model.predict(x_new)

    print('Predicted median house price for this block ($100,000s) ' + str(y_new[0]))
    test()



# Call the test function with the example parameters
test()

