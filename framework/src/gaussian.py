import numpy as np


num_distributions = 5  # number of total distributions
num_train = 4  # number of training distributions
num_points = 100  # number of points in a distribution
std_dev = 1  # devition
dim = 2  # dimension

def generate_gaussian(num_distributions = 5, num_train = 4, dim = 2, std_dev = 1):

    # step 1:  choose means in normal distribution
    means = np.random.normal(size=(num_distributions, dim))

    # Step 2 & 3: create gaussian distributions and split train and test distributions randomly
    indices = np.random.choice(range(num_distributions), size=num_train, replace=False)
    train_means = means[indices]
    test_mean = np.delete(means, indices, axis=0)[0]  # one left as test

    return train_means, test_mean


# step 4: generate train & test data
train_means, test_mean = generate_gaussian(num_distributions = 5, num_train = 4, dim = 2, std_dev = 1)
train_data = [np.random.normal(loc=mean, scale=std_dev, size=(num_points, dim)) for mean in train_means]
test_data = np.random.normal(loc=test_mean, scale=std_dev, size=(num_points, dim))

# create normal distribution as OT target distribution
target_distribution = np.random.normal(loc=np.zeros(dim), scale=std_dev, size=(num_points, dim))

# display data
train_means[0], test_mean, train_data[0][:2], test_data[:2], target_distribution[:2] 
