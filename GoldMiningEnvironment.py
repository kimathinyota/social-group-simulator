from src.Helper import *
from src.Interaction import *
import threading


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
        earnt = self.agent_earnt_last_round[victim] if victim in self.agent_earnt_last_round else 0
        return earnt/2

    def __init__(self, resource_total, minimum_mine):
        self.resource_total = resource_total
        self.minimum_mine = minimum_mine
        self.average_number_of_interactions_in_first_window = 20

        self.helping_funds = round(self.minimum_mine * 0.25)

        self.requested_interactions = []

        self.lock_to_access_possible_interactions = threading.Lock()
        self.agent_to_possible_interactions = {}
        self.confirmed_interactions = []

        self.interaction_types = [Friendship, Mentorship, Help, Theft]

        # Agent ->  {Friendship:[], Mentorship:[], Helps:[], Theft:[]}
        self.agent_to_all_interactions = {}

        # Social Hierarchy: Agent -> Wealth
        self.agent_to_wealth = {}

        self.agent_earnt_last_round = {}

        # Agent -> Number of rounds
        self.agent_to_rounds = {}

    def add_agent(self, agent):

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
                if i.is_bidirectional:
                    i = self.construct_interaction(agent, a, type)
                    self.agent_to_possible_interactions[a][type] = i
                    interactions_for_type.append(i)
                interactions[type] = interactions[type] + interactions_for_type if interactions[type] is not None \
                    else interactions_for_type

        self.agent_to_rounds[agent] = 0
        self.agent_to_wealth[agent] = 0
        self.agent_to_all_interactions[agent] = all_interactions
        self.agent_to_possible_interactions[agent] = interactions
        self.lock_to_access_possible_interactions.release()

    def get_possible_interactions(self, agent):
        self.lock_to_access_possible_interactions.acquire()
        interactions_copy = self.agent_to_possible_interactions[agent].copy()
        self.lock_to_access_possible_interactions.release()



        # weighted interactions


        for






        # notify agent about wealth

    def get_all_interactions(self, agent):
        if agent not in self.agent_to_all_interactions:
            return []
        interactions_dict = self.agent_to_all_interactions[agent]
        keys = list(self.agent_to_all_interactions.keys())

        interactions = []

        for key in keys:
            interactions += interactions_dict[key]

        return interactions

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























