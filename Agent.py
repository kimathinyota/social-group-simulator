from src.Helper import *
import time


class Agent:
    # d = dimension-list-of-raw-facet-scores [d1,d2,d3,d4,d5,d6]
    # where d ∈ [n,e,o,c,a] = [Neuroticism,Extraversion,Openness,Conscientiousness,Agreeableness]
    # 0 <= raw-facet-score <= 35

    def __init__(self, n, e, o, c, a, name):
        self.personality = personality_dictionary(n, e, o, c, a)
        self.social_group = None
        self.state = 'F'
        self.interacting_agents = []
        self.agents_status = {}
        self.name = name

    def set_social_group(self, social_group):
        self.social_group = social_group

    def action(self):
        return action_from_personality(self.personality)

    def prominence_agent(self, agent):
        if agent not in self.agents_status:
            return 0

        if len(self.agents_status) == 1:
            return 1

        tot = 0
        agent_prominence = self.agents_status[agent]['p']
        for a in self.agents_status:
            if a != agent and agent_prominence > self.agents_status[a]['p']:
                tot += 1

        return tot/len(self.agents_status)

    def status_agent(self, agent):
        if agent not in self.agents_status:
            return 0
        influence = self.agents_status[agent]['i']
        prominence = self.prominence_agent(agent)
        respect = self.agents_status[agent]['r']
        return influence+prominence+respect

    def interact_accept(self, agent):
        if self.state == 'B':
            return False

        similarity = 1
        if agent in self.agents_status:
            agent_personality = self.agents_status[agent]['a']
            similarity = personality_similarity(self.personality, agent_personality)

        extraversion = dimension_percentage(self.personality, 'E')
        agreeableness = dimension_percentage(self.personality, 'A')

        score = 2*(similarity*2 + extraversion*2 + similarity)
        random_score = apply_prob_distribution(score, 55, 7.5, 7)
        if random_score < 3.5:
            return False

        return True

    def request_interaction(self):
        if self.state == 'B':
            return False
        extraversion = dimension_percentage(self.personality, 'E')
        openness = dimension_percentage(self.personality, 'O')

        score = openness * 5 + extraversion * 5
        random_score = apply_prob_distribution(score, 55, 7.5, 7)
        if random_score < 3.5:
            return False

        return True

    def start_interaction(self, agent):
        self.state = 'B'
        self.interacting_agents.append(agent)
        self.interaction_start_time = time.time()

    def end_interaction(self):
        self.state = 'F'
        if len(self.interacting_agents) == 0:
            pass

        for agent in self.interacting_agents:
            action = agent.action()
            respect_observed = personality_facet_similarity(action, self.social_group.wanted_group_values,
                                                            self.social_group.unwanted_group_values)
            influence_observed = personality_facet_similarity(action, self.social_group.dominant_values,
                                                              self.social_group.non_dominant_values)

            if agent in self.agents_status:
                status_old = self.agents_status[agent]
                respect_old = status_old['r']
                influence_old = status_old['i']
                prominence_old = status_old['p']
                action_old = status_old['a']

                weight = logistic_update_status_weight(prominence_old, 0.8, 0.1, 20)
                respect_new = weight*respect_old + respect_observed*(1-weight)
                influence_new = weight * influence_old + influence_observed * (1 - weight)
                prominence_new = prominence_old + 1
                action_new = estimate_personality(action_old, action, weight)

                status_new = {'r': respect_new, 'i': influence_new, 'p': prominence_new, 'a': action_new}

                self.agents_status[agent] = status_new
            else:
                status_new = {'r': respect_observed, 'i': influence_observed, 'p': 1, 'a': action}
                self.agents_status[agent] = status_new

        self.interacting_agents = []
        self.interaction_start_time = None

    def run(self, social_group, lock):
        social_group.add_agent(self)
        #print("Agent " + self.name + ": Starting"

        while True:
            if self.state == 'F':
                attempt_interaction = self.request_interaction()
                if attempt_interaction == True:

                    lock.acquire()
                    found_agent = social_group.find_agent(self)
                    lock.release()
                    if found_agent != None:
                        social_group.start_interactions(self, found_agent)
                        time.sleep(2)

            else:

              social_group.end_interaction(self)


def random_agent(name):
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)


#n, e, o, c, a
def get_best_agent(name, wanted_values, unwanted_values):
    dimensions = ['N','E','O','C','A']
    bigfive = []
    for d in dimensions:
        temp = []
        for n in range(1,7):
            f = d + str(n)
            if f in wanted_values:
                temp.append(random.randrange(25,35))
            elif f in unwanted_values:
                temp.append(random.randrange(1,10))
            else:
                temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)

















