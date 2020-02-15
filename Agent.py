from src.Helper import *
import time
import uuid
from src.Interaction import *


class Competency:

    def random_competency(self, accuracy):
        return Competency(accuracy_value(self.mining_skill,accuracy),
                          accuracy_value(self.appraisal_skill,accuracy))

    def contains_greater_competency_skill(self, competency):
        return self.mining_skill > competency.mining_skill or self.appraisal_skill > competency.appraisal_skill

    def __init__(self, mining_skill, appraisal_skill):
        self.mining_skill = mining_skill
        self.appraisal_skill = appraisal_skill


class Agent:

    # Uses HEXACO personality model
    def __init__(self, name, competency, h, e, x, a, c, o):

        self.name = name
        self.id = uuid.uuid1()

        self.competency = competency

        self.number_of_interactions = 0

        # creates map from personality facets (e.g. 'X1' or 'C3') to score
        self.personality_template = HexacoPersonality()
        self.personality = self.personality_template.get_personality(h,e,x,a,c,o)

        # assigned social group
        self.social_group = None

        # is agent free (F) or busy (B) for interactions
        self.state = 'F'

        self.wealth = 0

        self.interactions = {}

        # list of all agents this agent is currently interacting with
        self.interacting_agents = []

        # AllInteractions = { Friendship, Mentorship, Help, Theft}
        # CurrentInteractions = { Friendship, Mentorship, Help, Theft}
        # memory of the agent - stores information about other agents
        # Agent -> Wealth, Competency Predictions, CurrentInteractions, AllInteractions

        self.agent_information = {}

    def get_number_of_thefts(self):
        thefts = self.interactions[Theft].copy()
        num = 0
        for t in thefts:
            if t.proactive_agent == self:
                num += 1
        return num

    # returns n.o times caught stealing
    def number_of_times_caught(self, agent):
        interactions = self.agent_information[agent]["AllInteractions"][Theft].copy()
        all_theft_count = 0
        for i in interactions:
            if i.proactive_agent == agent:
                all_theft_count += 1
        return all_theft_count

    def number_of_times_stolen_from_me(self, agent):
        thefts = self.interactions[Theft].copy()
        num = 0
        for t in thefts:
            if t.proactive_agent == agent and t.reactive_agent == self:
                num += 1
        return num

    def set_social_group(self, social_group):
        self.social_group = social_group

    def __repr__(self):
        return self.name








    def action(self):
        # based on the personality it will generate an action by applying a probability distribution
        # to each personality facet score
        return action_from_personality(self.personality)

    # prominence is just a count of how many times this agent has interacted with the input agent
    # this will return the percentage of agents that it has a greater (or equal) comparative prominence
    def prominence_agent(self, agent):
        # never interacted with this agent
        if agent not in self.agents_status:
            return 0

        if len(self.agents_status) == 1:
            return 1

        # retrieve current prominence
        agent_prominence = self.agents_status[agent]['p']

        return self.prominence_percentage(agent_prominence)

    # returns percentage of agents that this value is >= to
    def prominence_percentage(self, prominence):
        if len(self.agents_status) == 0:
            return 1
        tot = 0
        for a in self.agents_status:
            if prominence >= self.agents_status[a]['p']:
                tot += 1
        # find percentage
        return tot / (len(self.agents_status))

    # return status of agent as a score s, where 0 <= s <= 3
    def status_agent(self, agent):
        if agent not in self.agents_status:
            return 0
        influence = self.agents_status[agent]['i']
        prominence = self.prominence_agent(agent)
        respect = self.agents_status[agent]['r']

        if influence is None or respect is None:
            return None
        return influence+prominence+respect

    # will use similarity, extraversion and agreeableness percentage to
    # determine whether this agent will accept the interaction request from input agent
    def interact_accept(self, agent):
        if self.state == 'B':
            return False

        # if this agent hasn't met input, it will just assume they are very similar
        similarity = 1

        # will find the estimated personality for the input agent based on their actions
        # find cosine similarity between this agent's personality and the input agent
        if agent in self.agents_status:
            agent_personality = self.agents_status[agent]['a']
            similarity = personality_similarity(self.personality, agent_personality)

        extraversion = dimension_percentage(self.personality, 'E')
        agreeableness = dimension_percentage(self.personality, 'A')

        # ratio of 2:2:1 for similarity:extraversion:agreeableness chosen ...
        # this ratio will define the importance in those factors in influencing the probability that this agent accepts

        # score will positively correlate with probability of the agent requesting an interacting
        score = 2*(similarity*2 + extraversion*2 + agreeableness)
        random_score = apply_prob_distribution(score, 55, 7.5, 7)
        if random_score < 3.5:
            return False

        return True

    # will use openness and agreeableness percentage to
    # determine whether this agent will decide to request an interaction
    def request_interaction(self):
        if self.state == 'B':
            return False
        extraversion = dimension_percentage(self.personality, 'E')
        openness = dimension_percentage(self.personality, 'O')

        score = openness * 5 + extraversion * 5
        # score will positively correlate with probability of the agent requesting an interacting
        random_score = apply_prob_distribution(score, 55, 7.5, 7)
        if random_score < 3.5:
            return False

        return True

    # appraisal = difference in status after last interaction with the agent
    def appraisal(self, agent):
        if agent not in self.agents_status:
            return 0

        p = self.agents_status[agent]['l']
        c = self.status_agent(agent)

        r = c - p

        return r

    def start_interaction(self, agent):
        self.state = 'B'
        self.interacting_agents.append(agent)
        self.number_of_interactions += 1

    def request_appraisal(self, agent):
        if agent not in self.agents_status:
            return 0
        return self.agents_status - 1

    def observed_respect(self, action):
        # respect is calculated using the similarity between the agent's action and a 'high-status'
        # action based on the wanted and unwanted personality facets
        return personality_facet_similarity(action, self.social_group.wanted_group_values,
                                                            self.social_group.unwanted_group_values)

    def observed_influence(self, action):
        # influence is calculated using the similarity between the agent's action and a 'dominant'
        # action based on the dominant and non-dominant personality facets
        return personality_facet_similarity(action, self.social_group.dominant_values,
                                            self.social_group.non_dominant_values)

    # will end an interaction and update its current view of the statuses of the agents it interacted with
    def end_interaction(self):

        if len(self.interacting_agents) == 0:
            self.state = 'F'
            return

        # these are the agents it is interacting with
        for agent in self.interacting_agents:

            # get the observed action of the current agent
            action = self.social_group.get_action_during_interaction(self, agent)

            respect_observed = self.observed_respect(action)

            influence_observed = self.observed_influence(action)

            # check if agent has interacted with this agent before
            if agent in self.agents_status:
                # get current values that make up the current agent's status
                status_old = self.agents_status[agent]
                respect_old = status_old['r']
                influence_old = status_old['i']
                prominence_old = status_old['p']

                # action_old is essentially an estimate of the current agent's personality
                action_old = status_old['a']

                # update old values with current observations using below weight
                weight = logistic_update_status_weight(prominence_old, 0.8, 0.1, 20)

                if respect_old is None:
                    respect_new = respect_observed
                    respect_old = respect_observed
                else:
                    respect_new = weight * respect_old + respect_observed * (1 - weight)

                if influence_old is None:
                    influence_new = influence_observed
                    influence_old = influence_observed
                else:
                    influence_new = weight * influence_old + influence_observed * (1 - weight)

                last_status = respect_old + influence_old + self.prominence_percentage(prominence_old)

                prominence_new = prominence_old + 1

                if action is not None:
                    action_new = estimate_personality(action_old, action, weight)

                status_new = {'r': respect_new, 'i': influence_new, 'p': prominence_new, 'a': action_new, 'l': last_status }

                self.agents_status[agent] = status_new

            else:
                last_status = respect_observed + influence_observed + self.prominence_percentage(1)
                status_new = {'r': respect_observed, 'i': influence_observed, 'p': 1, 'a': action, 'l': last_status}
                self.agents_status[agent] = status_new

        self.interacting_agents = []
        # now free for interactions
        self.state = 'F'

    # agent running
    def run(self, social_group, lock):
        social_group.add_agent(self)
        while True:
            if self.state == 'F':
                # determine whether agent should request an interaction
                attempt_interaction = self.request_interaction()
                if attempt_interaction == True:
                    # agent decided to request interaction so it will use social_group.find_agent to do this

                    lock.acquire()
                    #found_agent = social_group.find_agent(self)

                    isFound = social_group.find_and_start_interactions(self)

                    if isFound == True:
                        time.sleep(0.25)

                    lock.release()

                    #if found_agent != None:
                    #    social_group.start_interactions(self, found_agent)
                    #    time.sleep(2)
            else:
                social_group.end_interaction(self)









def get_best_agent(name, wanted_values, unwanted_values):
    dimensions = ['N','E','O','C','A']
    bigfive = []
    for d in dimensions:
        temp = []
        for n in range(1,7):
            f = d + str(n)
            if f in wanted_values:
                temp.append(random.randrange(25, 35))
            elif f in unwanted_values:
                temp.append(random.randrange(1, 10))
            else:
                temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)










