from src.Helper import *
import time
import uuid
from src.Interaction import *


class Competency:

    def percentage(self):
        a = self.appraisal_skill
        m = self.mining_skill
        return (a + m)/2

    def percentage_difference(self, competency):
        cpr = self.percentage()
        crp = competency.percentage()
        return (1/2) * (1 + cpr - crp)

    def increase_mining_skills(self, percentage_increase):
        self.mining_skill *= percentage_increase
        if self.mining_skill > 1:
            self.mining_skill = 1
        elif self.mining_skill < 0:
            self.mining_skill = 0

    def increase_appraisal_skills(self, percentage_increase):
        self.appraisal_skill *= percentage_increase
        if self.appraisal_skill > 1:
            self.appraisal_skill = 1
        elif self.mining_skill < 0:
            self.mining_skill = 0

    def better_skills(self, competency):
        skills = []

        m = None
        if self.mining_skill > competency.mining_skill:
            skills.append("mining")
            m = (self.mining_skill - competency.mining_skill)

        a = None
        if self.appraisal_skill > competency.appraisal_skill:
            skills.append("appraisal")
            a = (self.appraisal_skill - competency.appraisal_skill)

        return skills, m, a

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

        # assigned environment
        self.environment = None

        # is agent free (F) or busy (B) for interactions
        self.state = 'F'

        self.wealth = 0

        self.interactions = {}

        # list of all agents this agent is currently interacting with
        self.interacting_agents = []

        self.pending_interaction_requests = []
        self.is_busy = False

        self.is_interacting = None
        self.is_mining = None

        self.is_running = None

        self.current_round = 0



        # Interactions = { Friendship, Mentorship, Help, Theft}
        # Agent -> Interactions -> Type -> ([Agent is Proactive], [Agent is Reactive])

        # memory of the agent - stores information about other agents
        # NoRounds = number of shared interaction rounds
        # Agent -> Wealth, Personality, Competency, Interactions, NoRounds

        self.agent_information = {}

    def get_number_of_thefts(self):
        thefts = self.interactions[Theft].copy()
        return len(thefts[0])

    # returns n.o times caught stealing
    def number_of_times_caught(self, agent):
        # Agent will be the proactive one
        return len(self.agent_information[agent]["Interactions"][Theft][0])

    def number_of_times_help(self, agent):
        return len(self.agent_information[agent]["Interactions"][Help][0])

    def increase_wealth(self, amount):
        self.wealth += amount
        self.environment.notify_wealth_increase(amount,self)

    def number_of_times_stolen_from_me(self, agent):
        reactive_thefts = self.interactions[Theft][1].copy()
        # Stolen from me so search the theft interactions where this agent is reactive
        num = 0
        for t in reactive_thefts:
            if t.is_present(agent):
                num += 1
        return num

    def set_social_group(self, environment):
        self.environment = environment

    def __repr__(self):
        return self.name

    def number_of_friendship_interactions(self,agent):
        friendship = self.interactions[Friendship].copy()
        f = 0
        for a in (friendship[0] + friendship[1]):
            if a.is_present(agent):
                f += 1
        return f

    def number_of_mentorship_interactions(self, agent):
        mentorship = self.interactions[Mentorship].copy()
        mp, mr = 0, 0
        for a in mentorship[0]:
            if a.is_present(agent):
                mp += 1

        for a in mentorship[1]:
            if a.is_present(agent):
                mr += 1
        return mr, mp

    def number_of_help_interactions(self, agent):
        help = self.interactions[Help].copy()
        mp, mr = 0, 0
        for a in help[0]:
            if a.is_present(agent):
                mp += 1

        for a in help[1]:
            if a.is_present(agent):
                mr += 1
        return mr, mp

    def number_of_theft_interactions(self, agent):
        theft = self.interactions[Theft].copy()
        mp, mr = 0, 0
        for a in theft[0]:
            if a.is_present(agent):
                mp += 1

        for a in theft[1]:
            if a.is_present(agent):
                mr += 1
        return mr, mp

    def forgiveness(self):
        return self.personality_template.facet_percentage(["A1"],self.personality)

    def friend(self, agent):
        # A = number of friend interactions between them
        A = self.number_of_friendship_interactions(agent)

        # personality = appraised personality(agent)
        personality = self.agent_information[agent]["Personality"]
        # B = similar personality
        B = self.personality_template.personality_similarity(self.personality, personality)
        # C = number of help interactions between them
        C = self.number_of_help_interactions(agent)[0]
        # D = number of theft interactions between them
        D = self.number_of_theft_interactions(agent)[0]
        # E = number of caught thefts
        E = self.number_of_times_caught(agent)
        # F = number of mentorship interactions between them
        F = self.number_of_mentorship_interactions(agent)[0]
        # G = number of times agent has helped
        G = self.number_of_times_help(agent)

        # Variable priority: D > A > C > B >= F > G >= E
        # n = number of interaction rounds
        n = self.agent_information[agent]["NoRounds"]
        # p = number of agents
        p = len(self.agent_information)
        # f = forgiveness
        f = self.forgiveness()

        return 0.5 + (1/40)(2*B + (1/n)(10*A + 5*C + 2*F + G) - (1-0.9*f)(17*D/n + E/(p*n)) )

    def stealing_aversion(self):
        a = self.personality_template.dimension_percentage(self.personality,"A")
        h = self.personality_template.dimension_percentage(self.personality,"H")
        return (a + h)/2

    def stealing_from_aversion(self, agent):
        friend = self.friend(agent)
        av = self.stealing_aversion()
        if friend >= 0.5:
            return math.pow(av, (7/13) - av * (friend - 0.5))
        return math.pow(av, 1 - av * (friend - 0.5))

    def help_probability(self, agent):
        return 0.5 * self.stealing_from_aversion(agent)

    def theft_probability(self, agent):
        return 1 - self.stealing_from_aversion(agent)

    def interact_probability(self):
        o = self.personality_template.dimension_percentage(self.personality, "O")
        e = self.personality_template.dimension_percentage(self.personality, "X")
        return (o + 5*e) / 6

    def teachable(self, personality):
        fsm = self.personality_template.max_facet_score

        o2, o3 = personality['O2'], personality['O3']
        e1, e2, e3 = personality['E1'], personality['E2'], personality['E3']

        c = self.personality_template.dimension_percentage(personality,'C')
        o = (o2 + o3)/2*fsm
        e = (e1 + e2 + e3) / 3 * fsm

        return (1/6) * (2 + o + 3*c - 2*e)

    def need_mentor(self, mentor, mentoring_cost):
        competency = self.agent_information[mentor]["Competency"]
        cd = competency.percentage_difference(self.competency)
        cost = mentoring_cost/(self.wealth if (self.wealth == 0 or mentoring_cost > self.wealth) else 1)
        return cd*(1-cost)

    def accept_mentor_probability(self, mentor, mentoring_cost):
        p = (1/5) * (2*(self.teachable(self.personality) + self.need_mentor(mentor,mentoring_cost)) + self.friend(mentor))
        return p

    def accept_mentee_probability(self, mentee):
        personality = self.agent_information[mentee]["Personality"]
        p = (1/3) * (self.teachable(personality) + self.friend(mentee))
        return p

    def accept_friend_probability(self, friend):
        p = self.friend(friend)
        return p

    def accept_interaction(self, interaction):
        if not interaction.is_present(self):
            return None

        if self.is_busy:
            self.pending_interaction_requests.append(interaction)
            return None

        # Agent not busy and so can deal with request

        other = interaction.other_agent(self)

        # only Friendship and Mentorship interactions require acceptance
        if isinstance(interaction,Friendship):
            accept = self.accept_friend_probability(other)
            return random_boolean_variable(accept)
        elif isinstance(interaction,Mentorship):
            accept = self.accept_mentor_probability(other) if self == interaction.reactive_agent else self.accept_mentee(other)
            return random_boolean_variable(accept)

        return None

    def request_interaction_probability(self, interaction):
        other = interaction.other_agent(self)
        if isinstance(interaction, Friendship):
            request = self.accept_friend_probability(other)
            return random_boolean_variable(request)
        elif isinstance(interaction, Mentorship):
            request = self.accept_mentor_probability(
                other) if self == interaction.reactive_agent else self.accept_mentee(other)
            return random_boolean_variable(request)
        elif isinstance(interaction, Theft):
            request = self.theft_probability(other)
            return random_boolean_variable(request)
        elif isinstance(interaction, Help):
            request = self.help_probability(other)
            return random_boolean_variable(request)
        return None

    def interaction_happened(self, interaction):
        # agent is notified of a successful interaction involving it
        if interaction in self.pending_interaction_requests:
            self.pending_interaction_requests.remove(interaction)
        if not interaction.is_success:
            return None
        # only successful interactions (both accept) are stored
        tup = self.interactions[type(interaction)]
        if interaction.proactive_agent == self:
            tup[0].append(interaction.copy())
        else:
            tup[1].append(interaction.copy())

        self.interactions[type(interaction)] = tup

    def notify_interaction(self, interaction):
        if not interaction.is_success:
            return None
        # only successful interactions (both accept) are stored
        other_agents = interaction.other_agent(self)
        for agent in other_agents:
            info = self.agent_information[agent]
            tup = info["Interactions"][type(interaction)]
            if agent == interaction.proactive_agent:
                tup[0].append(interaction.copy())
            else:
                tup[1].append(interaction.copy())
            self.agent_information[agent] = info

    def start_interacting(self):
        self.current_round += 1
        self.is_mining = False
        self.is_interacting = True
        self.is_busy = False

    def stop_interacting(self):
        self.is_busy = True
        self.is_interacting = False
        self.pending_interaction_requests.clear()

    def start_mining(self):
        self.is_mining = True

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.is_interacting:
                # agent is supposed to be interacting with other agents
                will_interact = random_boolean_variable(self.interact_probability())
                if will_interact:
                    # Agent decided to interact so will now request interaction options from environment
                    interactions, weights = ResourceMiningEnvironment.get_requestable_interactions(self.environment,self)
                    choice = random.choices(population=interactions, weights=weights, k=1)[0]
                    # Now agent will request interaction
                    choice.request(self)

                # Agent will now respond to up to two received interactions
                self.is_busy = True
                respond_to = []
                for i in range(min(2,len(self.pending_interaction_requests))):
                    respond_to.append(self.pending_interaction_requests[i])

                for interaction in respond_to:
                    interaction.respond(self)




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










