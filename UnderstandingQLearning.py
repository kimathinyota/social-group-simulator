import numpy as np

# Define the states
location_to_state = {'L1': 0,
                     'L2': 1,
                     'L3': 2,
                     'L4': 3,
                     'L5': 4,
                     'L6': 5,
                     'L7': 6,
                     'L8': 7,
                     'L9': 8
                     }

# Define the actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

rewards = np.array( [[0,1,0,0,0,0,0,0,0],
                     [1,0,1,0,1,0,0,0,0],
                     [0,1,0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,1,0,0],
                     [0,1,0,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0,0],
                     [0,0,0,1,0,0,0,1,0],
                     [0,0,0,0,1,0,1,0,1],
                     [0,0,0,0,0,0,0,1,0]] )

state_to_location = dict((state,location) for location,state in location_to_state.items())

# Initialise the parameters
gamma = 0.75 # Discount factor
alpha = 0.9 # Learning rate



# get_optimal_route() will
# take 2 arguments: starting location and end location in the warehouse
# return optimal route for reaching the end location from the starting location
# in form of ordered list containing the letters


def get_optimal_route(start_location, end_location):
    # Initialising Q-values
    Q = np.array(np.zeros([9,9]))

    # Copy rewards matrix to new matrix
    rewards_copy = np.copy(rewards)

    ending_state = location_to_state[end_location]
    # Priority of ending state will be set to the highest one
    rewards_copy[ending_state,ending_state] = 999

    for i in range(1000):
        # Pick random state
        current_state = np.random.radint(0,9)

        playable_actions = []

        # Iterate through copy of rewards matrix to get actions > 0
        # these are the directly reachable states from current state
        for j in range(9):
            if rewards_copy[current_state,j] > 0:
                playable_actions.append(j)

        # pick an action randomly from the list of playable actions leading us to the next state
        next_state = np.random.choice(playable_actions)

        # Compute the temporal difference
        # The action here exactly refers to going to the next state
        TD = rewards_copy[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[
            current_state, next_state]

        # Update the Q-Value using the Bellman equation
        Q[current_state, next_state] += alpha * TD

    # Initialize the optimal route with the starting location
    route = [start_location]

    # We don't know about the exact number of iterations needed to reach to the final
    # location hence while loop will be a good choice for iteratiing
    while (next_location != end_location):
        # Fetch the starting state
        starting_state = location_to_state[start_location]
        # Fetch the highest Q-value pertaining to starting state
        next_state = np.argmax(Q[starting_state,])
        # We got the index of the next state. But we need the corresponding letter.
        next_location = state_to_location[next_state]
        route.append(next_location)
        # Update the starting location for the next iteration
        start_location = next_location

    return route


