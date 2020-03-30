from src.Helper import *
import time
import uuid
from src.GoldMiningEnvironment import *

import threading


class Competency:

    def __repr__(self):
        x = "(" + str(round(self.mining_skill,3)) + "," + str(round(self.appraisal_skill,3)) + ")"
        return x

    def percentage(self):
        a = self.appraisal_skill
        m = self.mining_skill
        return (a + m)/2

    def percentage_difference(self, competency):
        a = self.appraisal_skill - competency.appraisal_skill
        m = self.mining_skill - competency.mining_skill
        return a,m

    def increase_mining_skills(self, amount):
        self.mining_skill += amount
        if self.mining_skill > 1:
            self.mining_skill = 1
        elif self.mining_skill < 0:
            self.mining_skill = 0

    def increase_appraisal_skills(self, amount):
        self.appraisal_skill += amount
        if self.appraisal_skill > 1:
            self.appraisal_skill = 1
        elif self.mining_skill < 0:
            self.mining_skill = 0

    def is_equivalent(self, competency):
        return competency.appraisal_skill == self.appraisal_skill and competency.mining_skill == self.mining_skill

    def update(self, mining_skill, appraisal_skill):
        self.mining_skill = mining_skill
        self.appraisal_skill = appraisal_skill

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

    def copy(self):
        return Competency(self.mining_skill,self.appraisal_skill)

    @staticmethod
    def random():
        return Competency(random.randrange(0,1000)/1000,random.randrange(0,1000)/1000)

    def random_competency(self, accuracy):
        return Competency(accuracy_value(100*self.mining_skill,accuracy)/100,
                          accuracy_value(100*self.appraisal_skill,accuracy)/100)

    def contains_greater_competency_skill(self, competency):
        return self.mining_skill > competency.mining_skill or self.appraisal_skill > competency.appraisal_skill

    def __init__(self, mining_skill, appraisal_skill):
        self.mining_skill = mining_skill
        self.appraisal_skill = appraisal_skill


class Agent:

    # only copies name, competency and personality
    def copy(self):
        return Agent(self.name,self.competency.copy(),self.personality.copy(), self.generation_id)

    # Uses HEXACO personality model
    def __init__(self, name, competency, personality, generation_id=None):

        self.name = name
        self.id = uuid.uuid1()

        self.generation_id = generation_id
        # maintained when agent is copied
        if self.generation_id is None:
            self.generation_id = uuid.uuid1()

        self.competency = competency

        self.number_of_interactions = 0

        self.max_number_of_interactions = None
        self.current_number_of_interactions = None

        # creates map from personality facets (e.g. 'X1' or 'C3') to score
        self.personality_template = HexacoPersonality()
        self.personality = personality

        self.interact_earn_lock = threading.Lock()
        self.interaction_to_earned = []

        # assigned environment
        self.environment = None

        self.interact_lock = threading.Lock()

        # is agent free (F) or busy (B) for interactions
        self.state = 'F'

        self.access_wealth_lock = threading.Lock()
        self.wealth = 0

        self.access_interactions_lock = threading.Lock()
        self.interactions = {Friendship:[[],[]], Mentorship:[[],[]], Help:[[],[]], Theft:[[],[]]}

        # list of all agents this agent is currently interacting with
        self.interacting_agents = []

        self.pending_interaction_requests = []
        self.is_busy = False

        self.is_interacting = None
        self.has_stopped_interaction = None
        self.is_mining = None

        self.is_running = None

        self.access_pending_interactions_lock = threading.Lock()

        self.current_round = 0

        self.is_in_prison = None
        self.caught_thefts_in_this_round = []

        # Interactions = { Friendship, Mentorship, Help, Theft}
        # Agent -> Interactions -> Type -> ([Agent is Proactive], [Agent is Reactive])

        # memory of the agent - stores information about other agents
        # NoRounds = number of shared interaction rounds
        # Agent -> Wealth, Personality, Competency, Interactions, NoRounds

        self.access_agent_information_lock = threading.Lock()
        self.agent_information = {}

    def acquire_interact_lock(self):
        self.interact_lock.acquire()

    def release_interact_lock(self):
        self.interact_lock.release()

    def update_wealth(self, agent, new_wealth):
        self.access_agent_information_lock.acquire()
        self.agent_information[agent]["Wealth"] = new_wealth
        self.access_agent_information_lock.release()

    def update_personality(self, agent, new_personality):
        self.access_agent_information_lock.acquire()
        self.agent_information[agent]["Personality"] = new_personality
        self.access_agent_information_lock.release()

    def update_competency(self, agent, new_competency):
        self.access_agent_information_lock.acquire()
        self.agent_information[agent]["Competency"] = new_competency
        self.access_agent_information_lock.release()

    def add_interaction_earnings(self, interaction_id, earnings):
        self.interact_earn_lock.acquire()
        c = (self.current_round - 1)
        if len(self.interaction_to_earned) <= c:
            for i in range(len(self.interaction_to_earned), c+1):
                self.interaction_to_earned.append({})
        self.interaction_to_earned[c][str(interaction_id)] = earnings
        self.interact_earn_lock.release()

    def increment_no_rounds(self):
        self.access_agent_information_lock.acquire()
        self.current_round += 1
        for agent in self.agent_information:
            self.agent_information[agent]["NoRounds"] += 1
        self.access_agent_information_lock.release()

    def add_new_agent(self, agent, wealth, personality, competency):
        interactions = {Friendship: ([],[]), Mentorship: ([],[]), Help: ([],[]), Theft: ([],[]) }
        info = {"Wealth": wealth, "Personality": personality, "Competency": competency, "Interactions":interactions, "NoRounds": 1}
        self.access_agent_information_lock.acquire()
        self.agent_information[agent] = info
        self.access_agent_information_lock.release()
        ##print("Included agents for ", self, " is ", list(self.agent_information.keys()))

    def get_number_of_thefts(self):
        thefts = self.interactions[Theft].copy()
        return len(thefts[0])

    # returns n.o times caught stealing
    def number_of_times_caught(self, agent):
        # Agent will be the proactive one
        self.access_agent_information_lock.acquire()
        k = len(self.agent_information[agent]["Interactions"][Theft][0])
        self.access_agent_information_lock.release()
        return k

    def number_of_times_help(self, agent):
        self.access_agent_information_lock.acquire()
        k = len(self.agent_information[agent]["Interactions"][Help][0])
        self.access_agent_information_lock.release()
        return k

    def increase_wealth(self, amount, should_notify_all=True, should_display=True, should_notify_environment=True, interaction_id=None):
        self.access_wealth_lock.acquire()
        self.wealth += amount
        self.access_wealth_lock.release()
        self.environment.notify_wealth_increase(amount, self, should_notify_all=should_notify_all, should_display=should_display, should_notify_environment=should_notify_environment)
        if interaction_id is not None:
            self.add_interaction_earnings(interaction_id, amount)

    def number_of_times_stolen_from_me(self, agent):
        reactive_thefts = self.interactions[Theft][1].copy()
        # Stolen from me so search the theft interactions where this agent is reactive
        num = 0
        for t in reactive_thefts:
            if t.is_present(agent):
                num += 1
        return num

    def add_caught_theft_for_this_round(self, theft):
        self.caught_thefts_in_this_round.append(theft)

    def set_environment(self, environment):
        self.environment = environment
        self.environment.add_agent(self)

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
        self.access_agent_information_lock.acquire()

        info = self.agent_information[agent]

        # p = number of agents
        p = len(self.agent_information)
        self.access_agent_information_lock.release()

        # A = number of friend interactions between them
        A = self.number_of_friendship_interactions(agent)

        # personality = appraised personality(agent)
        personality = info["Personality"]

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
        n = info["NoRounds"]

        # f = forgiveness
        f = self.forgiveness()

        return 0.5 + (1/40)*(2*B + (1/n)*(10*A + 5*C + 2*F + G) - (1-0.9*f)*(17*D/n + E/(p*n)) )

    def caught_stealing(self):
        c = self.personality_template.dimension_percentage(self.personality, 'C')
        n = self.personality_template.dimension_percentage(self.personality, 'E')
        x = self.get_number_of_thefts()
        a = self.environment.get_minimum_thefts_to_maximise_theft_skill()
        # below: probability of being caught
        p = 0.8 - (0.4 * (1 + c) * x) / (a * (1 + n))
        p += 0.1
        return p

    def stealing_aversion(self):
        a = self.personality_template.dimension_percentage(self.personality,"A")
        h = self.personality_template.dimension_percentage(self.personality,"H")
        return (a + h)/2

    def stealing_from_aversion(self, agent):
        friend = self.friend(agent)
        av = self.stealing_aversion()
        p = math.pow(av, 1 - 4 * av * (friend - 0.5))
        p = 1 if p > 1 else p
        return p

    def help_probability(self, agent):
        hw = agent.wealth
        mw = self.wealth
        if mw < 1 or hw < 1:
            bd = max(1 - hw, 1 - mw)
            hw += bd
            mw += bd
        t = mw / hw
        willingness = self.stealing_from_aversion(agent)
        a = 0.265 + (1-0.265)*(t - 0.90)/(2.5-0.90)
        a = min(3, max(0, a))
        z = min(a*willingness, 1)
        return z

    def risk_aversion(self):
        r = 1 - self.personality_template.facet_percentage(['O2','O4'],self.personality)
        e = self.personality_template.dimension_percentage(self.personality,'E')
        return (r + e)/2

    def theft_cost(self):
        n = len(self.caught_thefts_in_this_round)
        r = self.risk_aversion()
        return (r + 2*(1 - math.pow(1 - self.caught_stealing(), n + 1)))/3

    def theft_gain(self, agent):
        ow = self.agent_information[agent]["Wealth"]
        mew = agent.wealth
        thresh = 10
        if ow <= 0:
            return 0

        if mew <= 0:
            dif = 1-mew
            mew += dif
            ow += dif

        if ow > mew:
            return 0.5 + (0.5/thresh)*(ow/mew - 1)

        return 0.5 - (0.5/thresh)*(mew/ow - 1)

    def theft_probability(self, agent):
        willingness = 1 - self.stealing_from_aversion(agent)
        gain = self.theft_gain(agent)
        cost = self.theft_cost()
        p = willingness * gain * cost
        return p

    def interact_probability(self):
        o = self.personality_template.dimension_percentage(self.personality, "O")
        e = self.personality_template.dimension_percentage(self.personality, "X")
        return (o + 3*e) / 4

    def teachable(self, personality):
        fsm = self.personality_template.max_facet_score
        o2, o3, h4 = personality['O2'], personality['O3'], personality['H4']
        e1, e2, e3 = personality['E1'], personality['E2'], personality['E3']
        c = self.personality_template.dimension_percentage(personality,'C')
        o = (o2 + o3 + h4)/(3*fsm)
        e = (e1 + e2 + e3) / (3 * fsm)

        return (1/4) * (1.5 + o + 1.5*c - 1.5*e)

    def good_teacher(self, personality):
        good_teacher = self.personality_template.facet_percentage(["A3", "A4", "H2", "H1"], personality)
        return good_teacher

    def good_mentor(self, mentor):
        # teachable, A3, A4, H2, H1
        self.access_agent_information_lock.acquire()
        personality = self.agent_information[mentor]["Personality"]
        self.access_agent_information_lock.release()
        good_teacher = self.good_teacher(personality)
        teachable = self.teachable(personality)
        return (1.5 * teachable + good_teacher) / 2.5

    def need_mentor(self, mentor, mentoring_cost):
        self.access_agent_information_lock.acquire()
        competency = self.agent_information[mentor]["Competency"]
        self.access_agent_information_lock.release()
        a, m = competency.percentage_difference(self.competency)
        cd = ((a + m) + 2)/4
        if mentoring_cost is not None and self.wealth > 0 and mentoring_cost <= self.wealth:
            cost = min(2.5*mentoring_cost/self.wealth, 1)
        else:
            cost = 1
        nm = cd*(1-cost)
        return nm

    def accept_mentor_probability(self, mentor, mentoring_cost):
        teachable = self.teachable(self.personality)
        good_mentor = self.good_mentor(mentor)
        p = 1/6 * (2.5*teachable + 2.5*good_mentor + self.friend(mentor))
        n = self.need_mentor(mentor,mentoring_cost)
        r = 2*n * p
        r = 1 if r > 1 else r
        return r

    def accept_mentee_probability(self, mentee):
        self.access_agent_information_lock.acquire()
        personality = self.agent_information[mentee]["Personality"]
        self.access_agent_information_lock.release()
        teachable = self.teachable(personality)
        good_mentor = self.good_teacher(self.personality)
        p = (1/4) * (2*teachable + good_mentor + self.friend(mentee))
        return p

    def accept_friend_probability(self, friend):
        p = self.friend(friend)
        return p

    def accept_interaction(self, interaction):
        if not interaction.is_present(self):
            return None

        if self.is_busy:
            self.access_pending_interactions_lock.acquire()
            self.pending_interaction_requests.append(interaction)
            self.access_pending_interactions_lock.release()

            return None

        # Does agent even wan't to interact ?
        should_interact = random_boolean_variable(self.interact_probability())
        if not should_interact:
            return None

        # Agent not busy and so can deal with request

        other = interaction.other_agent(self)

        # only Friendship and Mentorship interactions require acceptance
        if isinstance(interaction,Friendship):
            accept = self.accept_friend_probability(other)
            x = "Accept probability: " + str(accept)
            #print(x)
            return random_boolean_variable(accept)
        elif isinstance(interaction,Mentorship):
            e = self.environment.estimated_earnings(self)
            accept = self.accept_mentor_probability(other,e) if self == interaction.reactive_agent else self.accept_mentee_probability(other)
            x = "Accept mentor: " + str(accept)
            #print(x)
            return random_boolean_variable(accept)

        return None

    def request_interaction_probability(self, interaction):
        other = interaction.other_agent(self)
        if isinstance(interaction, Friendship):
            request = self.accept_friend_probability(other)
            return request
        elif isinstance(interaction, Mentorship):
            if self == interaction.reactive_agent:
                # this agent is the mentee
                # with mentoring, the mentor will take 25% of the money earnt in the current round
                e = self.environment.estimated_mentoring_cost(self)
                request = self.accept_mentor_probability(other,e)
            else:
                # this agent is the mentor
                request = self.accept_mentee_probability(other)
            return request
        elif isinstance(interaction, Theft):
            request = self.theft_probability(other)
            return request
        elif isinstance(interaction, Help):
            request = self.help_probability(other)
            return request
        return None

    def increment_number_of_interactions(self, interaction):
        if not (not interaction.is_request_bidirectional and interaction.reactive_agent == self) and interaction.is_success:
            self.current_number_of_interactions += 1

    def interaction_happened(self, interaction):
        # agent is notified of a successful interaction involving it
        # if agent can request this interaction --> i.e has not control
        self.access_pending_interactions_lock.acquire()
        if interaction in self.pending_interaction_requests:
            self.pending_interaction_requests.remove(interaction)
        self.access_pending_interactions_lock.release()
        if not interaction.is_success:
            return None
        # only successful interactions (both accept) are stored
        self.access_interactions_lock.acquire()
        tup = self.interactions[type(interaction)]
        if interaction.proactive_agent == self:
            tup[0].append(interaction.copy())
        else:
            tup[1].append(interaction.copy())

        self.interactions[type(interaction)] = tup
        self.access_interactions_lock.release()

    def notify_interaction(self, interaction):
        if not interaction.is_success:
            return False
        # only successful interactions (both accept) are stored
        other_agents = interaction.other_agents(self)
        for agent in other_agents:
            self.access_agent_information_lock.acquire()
            info = self.agent_information[agent]

            tup = info["Interactions"][type(interaction)]
            if agent == interaction.proactive_agent:
                tup[0].append(interaction.copy())
            else:
                tup[1].append(interaction.copy())
            self.agent_information[agent] = info
            self.access_agent_information_lock.release()
        return True

    def start_interacting(self, in_prison=False):
        self.caught_thefts_in_this_round = []
        self.increment_no_rounds()
        self.current_number_of_interactions = 0
        self.max_number_of_interactions = self.environment.get_max_number_of_interactions_each_round()
        self.is_mining = False
        self.is_interacting = True
        self.is_in_prison = in_prison
        self.is_busy = in_prison

    def stop_interacting(self):
        self.is_busy = True
        self.is_interacting = False

    def start_mining(self):
        # clear any risidual interactions - this only called when every agent has stopped interacting
        self.pending_interaction_requests.clear()
        self.is_mining = True

    def stop_mining(self):
        pass

    @staticmethod
    def random(name):
        return Agent(name,Competency.random(),HexacoPersonality().random_personality())

    def stop_running(self):
        self.is_running = False

    def run(self):
        self.is_running = True
        while self.is_running:
            if self.is_interacting:
                # agent is supposed to be interacting with other agents
                lim = round(self.environment.get_max_number_of_interactions_each_round())
                can_interact = self.current_number_of_interactions < lim

                k = random_boolean_variable(self.interact_probability())

                if not self.is_in_prison and can_interact and k:
                    # Agent decided to interact so will now request interaction options from environment
                    interactions, weights, average_probability = ResourceMiningEnvironment.get_requestable_interactions(self.environment,self)

                    if interactions is not None:
                        if len(interactions) > 0:
                            choice = random.choices(population=interactions, weights=weights, k=1)[0]
                            #x = str(self) + " Choice: " + str(choice) + " <-->  " + str( { interactions[i]: weights[i] for i in range(len(interactions))})
                            #print(x)
                            choice.request(self)

                # Agent will now respond to up to two received interactions
                self.is_busy = True
                respond_to = []
                self.access_pending_interactions_lock.acquire()
                for i in range(min(2,len(self.pending_interaction_requests))):
                    respond_to.append(self.pending_interaction_requests[i])
                self.access_pending_interactions_lock.release()

                for interaction in respond_to:
                    interaction.respond(self)
            self.has_stopped_interaction = not self.is_interacting


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










