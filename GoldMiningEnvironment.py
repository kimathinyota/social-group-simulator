from src.Helper import *
from src.Interaction import *
import threading

#TO DO: appraised competency and personality
class ResourceMiningEnvironment:

    def construct_interaction(self, proactive_agent, reactive_agent, interaction_type):
        if interaction_type == Friendship:
            return Friendship(proactive_agent,reactive_agent,self)
        elif interaction_type == Mentorship:
            return Mentorship(proactive_agent,reactive_agent,self)
        elif interaction_type == Help:
            return Help(proactive_agent, reactive_agent, self)
        elif interaction_type == Theft:
            return Theft(proactive_agent, reactive_agent, self)
        return None

    def theft_amount(self,victim):
        earnt = self.agent_earnings_last_round[victim] if victim in self.agent_earnings_last_round else 0
        earnt /= 2
        return int(earnt) if earnt > 0 else 0

    def max_number_of_interactions_for_each_agent(self):
        return self.max_number_of_different_interactions * (len(self.active_agents) - 1)

    def get_minimum_thefts_to_maximise_theft_skill(self):
        return max(round(3*0.25*(len(self.active_agents) - 1)), 5)

    def __init__(self, resource_total, minimum_mine):
        self.resource_total = resource_total
        self.minimum_mine = minimum_mine
        self.average_number_of_interactions_in_first_window = 20

        self.helping_funds = round(self.minimum_mine * 0.25)

        self.is_running = None

        self.requested_interactions = []

        self.lock_to_access_possible_interactions = threading.Lock()
        self.agent_to_possible_interactions = {}
        self.confirmed_interactions = []
        self.interactions_to_promised_exchanges = {}

        self.active_agents = []

        self.interaction_types = [Friendship, Mentorship, Help, Theft]
        num = 0
        for i in self.interaction_types:
            num += 1
            if not i.is_single_role:
                num += 1

        self.max_number_of_different_interactions = num

        # Agent ->  {Friendship:[], Mentorship:[], Helps:[], Theft:[]}
        self.agent_to_all_interactions = {}

        self.agent_earnings_last_round = {}

        self.agent_earnings_current_round = {}

        # Agent -> Number of rounds
        self.agent_to_rounds = {}

        self.added_agents = []


    def add_agent(self, agent):
        self.added_agents.append(agent)



    def include_agent(self, agent):
        interactions = {}
        all_interactions = {}
        self.lock_to_access_possible_interactions.acquire()
        for a in list(self.agent_to_wealth.keys()):
            for type in self.interaction_types:
                all_interactions[type] = []
                interactions_for_type = []
                i = self.construct_interaction(agent,a,type)
                # acquire lock
                self.agent_to_possible_interactions[a][type] = i
                interactions_for_type.append(i)
                if not i.is_single_role:
                    i = self.construct_interaction(agent, a, type)
                    self.agent_to_possible_interactions[a][type] = i
                    interactions_for_type.append(i)
                interactions[type] = interactions[type] + interactions_for_type if interactions[type] is not None \
                    else interactions_for_type

        self.agent_to_rounds[agent] = 0
        self.agent_to_wealth[agent] = 0
        self.agent_to_all_interactions[agent] = all_interactions
        self.agent_to_possible_interactions[agent] = interactions
        self.active_agents.append(agent)
        self.lock_to_access_possible_interactions.release()


    def get_agent_earnings_this_round(self, agent):
        return self.agent_earnings_current_round[agent]

    def get_requestable_interactions(self, agent):
        self.lock_to_access_possible_interactions.acquire()
        interactions_copy = self.agent_to_possible_interactions[agent].copy()
        self.lock_to_access_possible_interactions.release()

        requestable_interactions = []
        weights = []
        for interaction in interactions_copy:
            # cant repeat a confirmed interaction
            if interaction.is_requestable_by(agent) and interaction not in self.confirmed_interactions:
                requestable_interactions.append(interaction)
                p = agent.request_interaction_probability(interaction)
                weights.append(p)

        # convert weights into ratios
        sum = 0
        for w in weights:
            sum += w

        for i in range(len(weights)):
            weights[i] /= sum

        return requestable_interactions, weights

    def get_all_interactions(self, agent):
        if agent not in self.agent_to_all_interactions:
            return []
        interactions_dict = self.agent_to_all_interactions[agent]
        keys = list(self.agent_to_all_interactions.keys())

        interactions = []

        for key in keys:
            interactions += interactions_dict[key]

        return interactions

    def notify_wealth_increase(self, amount, agent):
        e = self.agent_earnings_current_round[agent]
        en = amount if e is None else (e+amount)
        self.agent_earnings_current_round[agent] = en

    def get_all_interactions_between(self, agent1, agent2):
        first_interactions = self.get_all_interactions(agent1)
        interactions = []
        for interaction in first_interactions:
            if interaction.contains(agent1,agent2):
                interactions.append(interaction)
        return interactions

    def appraisal(self, agent, agent_to_appraise):
        interactions = self.get_all_interactions_between(agent,agent_to_appraise)
        n = len(interactions)
        askill = agent.competency.appraisal_skill
        accuracy = askill * (0.2 + n/3*self.average_number_of_interactions_in_first_window)
        personality = agent_to_appraise.personality
        appraised_personality = agent_to_appraise.personality_template.generate_random_personality(personality,accuracy)
        appraised_competency = agent_to_appraise.competency.random_competency(accuracy)
        return {"personality": appraised_personality, "competency": appraised_competency}

    def notify_interaction(self, interaction):
        self.confirmed_interactions.append(interaction)
        if interaction.is_success:
            self.agent_to_all_interactions[interaction.proactive_agent][type(interaction)].append(interaction.copy())
            self.agent_to_all_interactions[interaction.reactive_agent][type(interaction)].append(interaction.copy())

    def notify_all_agents(self, interaction):
        for agent in self.active_agents.copy():
            agent.notify_interaction(interaction)

    def process_later(self, interaction, exchange):

        lst = self.interactions_to_promised_exchanges[interaction]
        lst = [] if lst is None else lst
        lst.append(exchange)
        self.interactions_to_promised_exchanges[interaction] = lst

    def process_now(self, exchange):
        exchange.process()

    def add_all_agents(self):
        for agent in self.added_agents:

    def run(self):
        self.is_running = True
        while self.is_running:
            # Before starting interaction stage: add agents
            for agent in self.added_agents:





class Token:

    def remove_from(self, agent):
        pass

    def add_to(self, agent):
        pass

    def __init__(self):
        pass


class ResourceToken(Token):

    def remove_from(self, agent):
        amount = -1 * self.amount
        return ResourceToken.change(amount, agent)

    def add_to(self, agent):
        amount = self.amount
        return ResourceToken.change(amount,agent)

    @staticmethod
    def change(amount, agent):
        agent.increase_wealth(amount)
        return True

    def __init__(self, amount):
        Token.__init__(self)
        self.amount = amount


class CompetencyToken(Token):

    def remove_from(self, agent):
        p = (100 - self.percentage) / 100
        return self.change_competency(p, agent)

    def change_competency(self,p, agent):
        for s in self.skill_list:
            if s == "appraisal":
                agent.competency.increase_appraisal_skills(p)
            else:
                agent.competency.increase_mining_skills(p)
        return True

    def add_to(self, agent):
        p = (self.percentage + 100)/100
        return self.change_competency(p,agent)

    def __init__(self, percentage, skill_list):
        Token.__init__(self)
        self.percentage = percentage
        self.skill_list = skill_list


class Exchange:

    def process(self):
        # remove lost_token from from_agent
        self.lost_token.remove_from(self.from_agent)

        # add gain_token to to_token
        self.gained_token.add_to(self.to_agent)


    # during exchange: from_agent loses lost_token and to_agent gains gained_token
    def __init__(self, from_agent, to_agent, lost_token, gained_token):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.lost_token = lost_token
        self.gained_token = gained_token
























