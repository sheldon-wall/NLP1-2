import numpy as np # Library for linear algebra and math utils
import pandas as pd # Dataframe library

import matplotlib.pyplot as plt # Library for plots
from utils import confidence_ellipse # Function to add confidence ellipses to charts
from utils import plot_confidence_ellipses

# Calculate the likelihood for each tweet
# for purposes of
data = pd.read_csv('bayes_features.csv')  # load the data file from the csv file
data.head(5)

# Plot the samples using columns 1 and 2 of the matrix

fig, ax = plt.subplots(figsize = (8, 8)) #Create a new figure with a custom size

colors = ['red', 'green'] # Define a color palete

# Color base on sentiment
ax.scatter(data.positive, data.negative,
           c=[colors[int(k)] for k in data.sentiment], s = 0.1, marker='*')  # Plot a dot for each tweet

# Custom limits for this chart
plt.xlim(-250,0)
plt.ylim(-250,0)

plt.xlabel("Positive") # x-axis label
plt.ylabel("Negative") # y-axis label
plt.show()

## ==========================================================================
## Using Confidence Ellipses to interpret Naive Bayes
## ==========================================================================

plot_confidence_ellipses(data)

## Now muck about with the data so that we can see the difference
data2 = data.copy() # Copy the whole data frame
# The following 2 lines only modify the entries in the data frame where sentiment == 1
data2.negative[data.sentiment == 1] =  data2.negative * 1.5 + 50 # Modify the negative attribute
data2.positive[data.sentiment == 1] =  data2.positive / 1.5 - 50 # Modify the positive attribute
plot_confidence_ellipses(data2)

