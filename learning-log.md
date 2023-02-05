# Week 0

02/04/2023
## Theory
### Cross-entropy Method (CEM)

Cross-entropy Method (CEM) is a Monte Carlo method for **importance sampling and optimization**. It's gradient free but model-based (as we are using a policy?)

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
