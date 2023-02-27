# Week 1

02/04/2023
## Theory
### Cross-entropy Method (CEM)

Cross-entropy Method (CEM) is a Monte Carlo method for **importance sampling and optimization**. It's gradient free but model-based (as we are using a policy?). We just learned a **tabular CEM**. It's an **evolutionary** algorithm and works extremely well.

https://www.youtube.com/watch?v=aUrX-rP_ss4&list=PLCTc_C7itk-GaAMxmlChrkPnGKtjz8hv1

1. Initialize a policy

Then we traing the policy 100 times with:
    1. Generate 250 episodes with the policy saving states, actions, and total reward for each episode
    2. Get episodes with 50th percentile rewards (keeping good episodes)
    3. now for each state across the good episodes, we are counting the number of each action. With the counts, we are generating P(a|s). We get a new policy
    4. based on the learning rate, we update our policy
    


### Code

1. We can apply a method on an axis in numpy.
2. all values in a numpy 2D array can be updated with arr[:, :] = some value
3. np.all, np.allclose can be used to verify values of an array
4. np.percentile gives us threshold of a percentile from a given array like values
5. plt.vlines can be used to show vertical lines
6. np.isfinite(new_policy).all() to check NaN or +-inf
7. IPython.display.clear_output can be used to update figures inside notebook
8. np.isclose is a handy tool to write quick unit tests

### CEM with scikit-learn deep learning (NOT finished yet)
1. MLPClassifier has a partial_fit method which retains weights. fit method reinitializes weights
2. We can record gym rendered videos with gym.wrappers.Monitor.

### Gym
Some environments have 200 tick limit. We can remove the time limit wrapper by using gym.make('envName').env (the base environment does not have any time limit)

### Faster Training with Joblib

We can parallelize espisode creationg using joblib! [More here](https://www.coiled.io/blog/sklearn-joblib-dask)


# Week 2

## Theory

### Difference between value iteration, policy iteration, and generalized policy iteration.

### Contraction Mapping:

Explanation of the notation is [here](https://www.youtube.com/watch?v=_DynXugXksU). The infinity norm just states the maximum value of a state in different value functions.

1. ||TV||inf = maximum value over states. 
2. ||TV - TU||inf = difference of maximum values over states of two different value functions. (the states can be different)
3. ||V - U||inf = maximum difference of the value functions calculated over same states, i.e. f(sNot) = V(sNot) - U(sNot). 

So, the contraction property says that the difference between maximum state values of two differnt policies are smaller than maximum difference policy of the policy (because why?). 

