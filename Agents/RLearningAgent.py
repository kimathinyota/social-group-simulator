import src.Helper as Helper
from src.Agents.Agent import Agent
import numpy as np
import random
import time


class RLearningAgent(Agent):

    def __init__(self, n, e, o, c, a, name):

        Agent.__init__(self, n, e, o, c, a, name)

        self.action_space = Helper.get_actions_space(1)

        self.Q = np.array(np.zeros([50,32]))

        self.is_training = True

        self.number_of_training_interactions = 50

        self.best_actions = []
        self.worst_actions = []

        self.wanted_facets = []
        self.unwanted_facets = []

        self.last_action = None
        self.last_personality = None

        self.gamma = 0.75 # Discount factor
        self.alpha = 0.9 # Learning rate

    def reward(self):
        reward = self.social_group.get_appraisal_during_interaction(self)
        if reward is None:
            return 0
        return reward

    def appraisal(self, agent):
        if self.is_training:
            return 0

        return super(RLearningAgent, self).appraisal(agent)

    def start_interaction(self, agent):
        super(RLearningAgent, self).start_interaction(agent)
        if self.number_of_interactions == self.number_of_training_interactions:
            self.complete_training()

    def action(self):
        if not self.is_training:
            # apply best and worst actions to personality
            for a in self.best_actions:
                self.personality = Helper.apply_action_to_personality(self.personality, a, 1)
            for a in self.worst_actions:
                self.personality = Helper.subtract_action_from_personality(self.personality,a, 1)
            return self.personality

        # while in training mode, each action will be randomly selected
        self.last_action = random.randrange(len(self.action_space))
        self.last_personality = self.personality
        # change personality slightly
        self.personality = Helper.apply_action_to_personality(self.personality, self.action_space[self.last_action], 4)

        return self.personality

    # return status of agent as a score s, where 0 <= s <= 3
    def status_agent(self, agent):
        if agent not in self.agents_status:
            return 0

        if self.is_training:
            # agent can only judge status on prominence
            return 3*self.prominence_agent(agent)

        return super(RLearningAgent, self).status_agent(agent)

    def complete_training(self):

        actions_indexes = self.Q.flatten().argsort()

        tot = len(actions_indexes)

        thresh = 10

        self.best_actions = []
        self.worst_actions = []

        print("Tot: " + str(tot) + " Thresh: " + str(thresh) )

        best_indexes = actions_indexes[(tot-thresh):tot:1]
        worst_indexes = actions_indexes[0:thresh:1]

        print("Best Actions: " + str(best_indexes))
        print("Worst Actions: " + str(worst_indexes))

        self.best_actions.append(self.action_space[best_indexes[0] % len(self.action_space)])
        self.worst_actions.append(self.action_space[worst_indexes[0] % len(self.action_space)])

        for index in best_indexes:
            action = self.action_space[index % len(self.action_space)]
            b = Helper.contains_arr(action,self.best_actions)
            if not b:
                self.best_actions.append(action)

        self.wanted_facets = Helper.retrieve_personality_facets(self.best_actions)
        print("WF: " + str(self.wanted_facets))

        for index in worst_indexes:
            action = self.action_space[index % len(self.action_space)]

            #b = (np.asarray(self.best_actions) == action).all(1).any
            #c = (np.asarray(self.worst_actions) == action).all(1).any

            b = Helper.contains_arr(action,self.best_actions)
            c = Helper.contains_arr(action, self.worst_actions)

            #print(str(Helper.retrieve_personality_facets([action])) + ": " + str(b == True) + " , " + str(c == True))

            if not b and not c:
                self.worst_actions.append(action)


        self.unwanted_facets = Helper.retrieve_personality_facets(self.worst_actions)

        print("Q: ")
        print(self.Q)

        print("UWF: " + str(self.unwanted_facets))

        self.is_training = False

    def end_interaction_training(self):
        action = self.last_action
        s = Helper.personality_score(self.last_personality, 10)
        s2 = Helper.personality_score(self.personality, 10)

        current = time.time()

        # reward is calculated by taking the average appraisal for a minimum of 50% of the  agent involved
        # reward will return none if not enough agents have completed the interaction
        # a timeout of 2.5s will be used to avoid any deadlocks

        reward = self.reward()
        while reward is not None and (time.time() - current) < 2.5:
            reward = self.reward()


        if reward is None:
            reward = 0



        # Compute the temporal difference

        td = reward + self.gamma * self.Q[s2][np.argmax(self.Q[s2,])]

        # Update the Q-Value using the Bellman equation
        self.Q[s][action] += self.alpha * td

        # evaluating status for each agent
        # agent is still learning social environment so can't measure respect and influence yet
        # agent can observe prominence

        # these are the agents it is interacting with
        for agent in self.interacting_agents:

            # get the observed action of the current agent
            action = self.social_group.get_action_during_interaction(self, agent)

            # check if agent has interacted with this agent before
            if agent in self.agents_status:
                # get current values that make up the current agent's status
                status_old = self.agents_status[agent]
                prominence_old = status_old['p']
                # action_old is essentially an estimate of the current agent's personality
                action_old = status_old['a']
                last_status = self.status_agent(agent)

                weight = Helper.logistic_update_status_weight(prominence_old, 0.8, 0.1, 20)

                prominence_new = prominence_old + 1

                action_new = Helper.estimate_personality(action_old, action, weight)

                status_new = {'r': None, 'i': None, 'p': prominence_new, 'a': action_new,
                              'l': last_status}

                self.agents_status[agent] = status_new

            else:
                last_status = 3*self.prominence_percentage(1)
                status_new = {'r': None, 'i': None, 'p': 1, 'a': action, 'l': last_status}
                self.agents_status[agent] = status_new

        self.interacting_agents = []
        # now free for interactions
        self.state = 'F'

    def observed_respect(self, action):
        # respect is calculated using the similarity between the agent's action and a 'high-status'
        # action based on the wanted and unwanted personality facets
        return Helper.personality_facet_similarity(action, self.wanted_facets,
                                                   self.unwanted_facets)

    def observed_influence(self, action):
        return self.observed_respect(action)

    def end_interaction(self):
        if len(self.interacting_agents) == 0:
            self.state = 'F'
            pass

        if self.is_training:
            self.end_interaction_training()
            return

        super(RLearningAgent, self).end_interaction()


def random_r_agent(name):
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(random.randrange(1, 34))
        bigfive.append(temp)

    return RLearningAgent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)


def zero_r_agent(name):
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(1)
        bigfive.append(temp)

    return RLearningAgent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)






