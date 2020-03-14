
import operator
from abc import ABC, abstractmethod
import threading

import time
from src.Helper import *
from src.DisplayServer import *
from src.Testing import *
from src.Analysis import *


# TODO: appraised competency and personality


class ResourceMiningEnvironment:

    def construct_interaction(self, proactive_agent, reactive_agent, interaction_type):
        if interaction_type == Friendship:
            return Friendship(proactive_agent, reactive_agent, self)
        elif interaction_type == Mentorship:
            return Mentorship(proactive_agent,reactive_agent, self)
        elif interaction_type == Help:
            return Help(proactive_agent, reactive_agent, self)
        elif interaction_type == Theft:
            return Theft(proactive_agent, reactive_agent, self)
        return None

    def theft_amount(self,victim):
        self.access_earnings_last_lock.acquire()
        if victim not in self.agent_earnings_last_round:
            self.access_earnings_last_lock.release()
            return 0
        earnt = self.agent_earnings_last_round[victim] / 2
        self.access_earnings_last_lock.release()
        return int(max(0.5*self.minimum_mine,earnt))

    def helping_amount(self, help):
        self.access_earnings_last_lock.acquire()
        earnt = self.agent_earnings_last_round[help] if help in self.agent_earnings_last_round else 0
        earnt *= 0.05
        self.access_earnings_last_lock.release()
        return int(max(0.25*self.minimum_mine, earnt))

    def estimated_mentoring_cost(self,mentee):
        cost = max(0.25*self.minimum_mine,1)
        if mentee in self.agent_earnings_last_round:
            self.access_earnings_last_lock.acquire()
            earnt = self.agent_earnings_last_round[mentee]
            self.access_earnings_last_lock.release()
            earnt *= 0.6
            if int(earnt) <= 0:
                rounds = mentee.current_round
                cost *= 1.5*rounds
            else:
                cost = earnt
        return int(cost)

    def max_number_of_interactions_for_each_agent(self):
        return self.max_number_of_different_interactions * (len(self.active_agents) - 1)

    def get_max_number_of_interactions_each_round(self):
        return int(math.pow(self.max_number_of_interactions_for_each_agent(),(1/3)))

    def get_minimum_thefts_to_maximise_theft_skill(self):
        return max(round(3*0.25*(len(self.active_agents) - 1)), 5)

    def get_competency_increase_percentage(self):
        return 0.15/self.max_number_of_interactions_for_each_agent()

    def __init__(self, resource_total, minimum_mine, number_of_rounds, should_test=False):

        self.number_of_rounds = number_of_rounds

        self.interaction_types = [Friendship, Mentorship, Help, Theft]

        self.analysis = Analysis()

        self.resource_total = resource_total
        self.minimum_mine = minimum_mine
        self.average_number_of_interactions_in_first_window = 20

        self.helping_funds = round(self.minimum_mine * 0.25)

        self.is_running = None

        self.requested_interactions = []

        self.lock_to_access_possible_interactions = threading.Lock()
        self.agent_to_possible_interactions = {}
        self.all_requested_interactions = []
        self.lock_requested_interactions = threading.Lock()

        self.confirmed_interactions = []

        self.access_promise_exchange_lock = threading.Lock()
        self.interactions_to_promised_exchanges = {}

        self.interaction_timer = None

        self.access_display_lock = threading.Lock()
        self.display_requests = []

        self.active_agents = []

        num = 0
        for i in self.interaction_types:
            num += 1
            if not i.is_single_role:
                num += 1

        self.max_number_of_different_interactions = num

        # Agent ->  {Friendship:[], Mentorship:[], Helps:[], Theft:[]}

        self.access_agent_to_all_interactions_lock = threading.Lock()
        self.agent_to_all_interactions = {}

        self.access_earnings_last_lock = threading.Lock()
        self.agent_earnings_last_round = {}

        self.access_agent_earning_current_lock = threading.Lock()
        self.agent_earnings_current_round = {}

        self.agent_earnings_after_mining = {}

        # Buffer that stores incoming agents that are to be added
        self.to_be_added_agents = []

        self.prison = []
        self.prison_lock = threading.Lock()
        self.prison_for_next_round = []

        self.agent_to_prison_sentence = {}

        self.time_per_interaction_session = None

        self.current_round = 0

        self.in_interaction_mode = False
        self.in_mining_mode = False
        self.in_before_interaction_mode = True

        self.agent_wealth_before_interaction_test = None
        self.agent_wealth_after_interaction_test = None
        self.agent_wealth_after_mining_test = None

        self.agent_competency_before_interaction_test = None
        self.agent_competency_after_interaction_test = None

        self.should_test = should_test

    def display(self, code, args):
        self.access_display_lock.acquire()
        self.display_requests.append(ServiceGUI.construct_request(code,args))
        self.access_display_lock.release()

    def get_non_imprisoned_agents(self):

        active_agents = self.active_agents.copy()
        for agent in self.prison:
            active_agents.remove(agent)
        return active_agents

    def add_to_prison_for_next_round(self, agent):
        self.prison_lock.acquire()
        self.agent_to_prison_sentence[agent] = 1 if agent not in self.agent_to_prison_sentence else \
            self.agent_to_prison_sentence[agent] + 1
        self.prison_lock.release()

    def set_time_for_interaction_session(self):
        self.time_per_interaction_session = (self.max_number_of_interactions_for_each_agent() / (7*2*2))/50

    def add_agent(self, agent):
        self.to_be_added_agents.append(agent)

    def include_agent(self, agent):
        all_interactions = {}
        possible_interactions = {}
        for t in self.interaction_types:
            possible_interactions[t] = []
            all_interactions[t] = []

        # Need to find and insert all of the possible interactions for the input agent into possible_interactions
        # Need to add extra possible interactions for other active agents based on this input agent
        # Need to notify other agents of this agents existence - send them appraisal information

        self.lock_to_access_possible_interactions.acquire()

        for a in self.active_agents:
            # send agent appraisal information to a
            personality, competency = self.get_appraised_personality_and_competency(a,agent)
            a.add_new_agent(agent,agent.wealth,personality,competency)

            personality, competency = self.get_appraised_personality_and_competency(agent, a)
            agent.add_new_agent(a,a.wealth,personality,competency)

            for type in self.interaction_types:
                interactions_for_type = []
                i = self.construct_interaction(agent,a,type)
                if i is not None:
                    interactions_for_type.append(i)
                    if not i.is_single_role:
                        i = self.construct_interaction(a, agent, type)
                        if i is not None:
                            interactions_for_type.append(i)

                # interactions_for_type contains all possible interactions of this type involving agents a and agent

                possible_interactions[type] += interactions_for_type
                self.agent_to_possible_interactions[a][type] += interactions_for_type

        self.agent_earnings_current_round[agent] = 0

        self.agent_to_possible_interactions[agent] = possible_interactions

        self.lock_to_access_possible_interactions.release()

        self.agent_to_all_interactions[agent] = all_interactions

        self.active_agents.append(agent)

    def get_agent_earnings_this_round(self, agent):
        self.access_agent_earning_current_lock.acquire()
        earnings = self.agent_earnings_current_round[agent]
        self.access_agent_earning_current_lock.release()
        return earnings

    def get_agent_earnings_after_mining(self, agent):

        earnings = self.agent_earnings_after_mining[agent]

        return earnings

    def get_requestable_interactions(self, agent):

        if agent in self.prison:
            return None, None, None

        self.lock_to_access_possible_interactions.acquire()
        interactions_copy = self.agent_to_possible_interactions[agent].copy()
        self.lock_to_access_possible_interactions.release()

        requestable_interactions = []
        weights = []

        total = 0
        n = 0

        high = 0

        for interaction_type in interactions_copy:
            # cant repeat a confirmed interaction
            for interaction in interactions_copy[interaction_type]:
                Interaction.initiate_interaction(interaction)
                neither_in_prison = interaction.proactive_agent not in self.prison and interaction.reactive_agent not in self.prison
                if neither_in_prison and interaction.is_requestable_by(agent) and not (interaction in self.confirmed_interactions):
                    requestable_interactions.append(interaction)
                    p = agent.request_interaction_probability(interaction)
                    if p > high:
                        high = p
                    total += p
                    n += 1
                    weights.append(p)

        average = total/n if n > 0 else 0

        # convert weights into ratios
        sum = 0
        for w in weights:
            sum += w

        for i in range(len(weights)):
            weights[i] /= sum

        return requestable_interactions, weights, high

    def get_all_interactions(self, agent):

        self.access_agent_to_all_interactions_lock.acquire()

        #THIS IS THE ERROR
        if agent not in self.agent_to_all_interactions:
            self.access_agent_to_all_interactions_lock.release()
            return []
        interactions_dict = self.agent_to_all_interactions[agent]

        keys = list(interactions_dict.keys())

        self.access_agent_to_all_interactions_lock.release()

        interactions = []

        for key in keys:
            interactions += interactions_dict[key]

        return interactions

    def notify_wealth_increase(self, amount, agent, should_notify_all=True, should_display=True, should_notify_environment=True):
        if should_display:
            self.display(3, [self.get_agent_to_wealth()])

        if not should_notify_environment:
            return None
        self.access_agent_earning_current_lock.acquire()
        e = self.agent_earnings_current_round[agent]
        en = amount if e is None else (e+amount)
        self.agent_earnings_current_round[agent] = en
        self.access_agent_earning_current_lock.release()
        if not should_notify_all:
            return None
        for a in self.active_agents:
            if a != agent:
                a.update_wealth(agent, agent.wealth)

    def get_all_interactions_between(self, agent1, agent2):
        first_interactions = self.get_all_interactions(agent1)
        interactions = []
        for interaction in first_interactions:
            if interaction.contains(agent1,agent2):
                interactions.append(interaction)
        return interactions

    def get_appraised_personality_and_competency(self, agent, agent_to_appraise):
        interactions = self.get_all_interactions_between(agent,agent_to_appraise)
        n = len(interactions)
        askill = agent.competency.appraisal_skill
        accuracy = askill * (0.2 + n/3*self.average_number_of_interactions_in_first_window)
        personality = agent_to_appraise.personality
        appraised_personality = agent_to_appraise.personality_template.generate_random_personality(personality,accuracy)
        appraised_competency = agent_to_appraise.competency.random_competency(accuracy)
        return appraised_personality, appraised_competency

    def notify_interaction(self, interaction):
        self.confirmed_interactions.append(interaction)
        p = interaction.proactive_agent.personality_template.personality_similarity(interaction.proactive_agent.personality,interaction.reactive_agent.personality)
        a = helper.similarity(interaction.proactive_agent.competency.appraisal_skill,interaction.reactive_agent.competency.appraisal_skill)
        m = helper.similarity(interaction.proactive_agent.competency.mining_skill,interaction.reactive_agent.competency.mining_skill)

        self.analysis.add_interaction(interaction,(p,m,a),self.current_round)
        self.access_agent_to_all_interactions_lock.acquire()

        if interaction.is_success:
            self.agent_to_all_interactions[interaction.proactive_agent][type(interaction)].append(interaction.copy())
            self.agent_to_all_interactions[interaction.reactive_agent][type(interaction)].append(interaction.copy())

        self.access_agent_to_all_interactions_lock.release()

        prison_copy = self.prison.copy()
        all_agents = prison_copy + self.get_non_imprisoned_agents()
        self.display(1,[[interaction.copy()],all_agents,prison_copy])

    def notify_all_agents(self, interaction):
        for agent in self.active_agents.copy():
            agent.notify_interaction(interaction)

    def process_later(self, interaction, exchange):
        lst = []
        self.access_promise_exchange_lock.acquire()
        if interaction in self.interactions_to_promised_exchanges:
            lst = self.interactions_to_promised_exchanges[interaction]
        lst = [] if lst is None else lst
        lst.append(exchange)
        self.interactions_to_promised_exchanges[interaction] = lst
        self.access_promise_exchange_lock.release()

    def process_now(self, exchange):
        exchange.process()

    def include_all_agents(self):
        for agent in self.to_be_added_agents:
            self.include_agent(agent)
        self.analysis.include_agents(self.to_be_added_agents)
        self.to_be_added_agents.clear()

    def estimated_earnings(self, agent):
        last_earnt = 1.2*self.agent_earnings_last_round[agent] if agent in self.agent_earnings_last_round else 0

        if last_earnt <= 0:
            last_earnt = 1.2*self.minimum_mine

        return int(last_earnt)

    def start_interaction_timer(self):
        self.interaction_timer = time.time()

    def get_agent_to_wealth(self):
        agent_to_wealth = {}
        for agent in self.active_agents:
            agent_to_wealth[agent] = agent.wealth
        return agent_to_wealth

    def start_interactions(self):
        if self.should_test:
            self.agent_wealth_before_interaction_test = {agent: agent.wealth for agent in self.active_agents}
            self.agent_competency_before_interaction_test = {agent: agent.competency.copy() for agent in self.active_agents}
        self.in_mining_mode = False
        self.in_before_interaction_mode = False
        self.in_interaction_mode = True
        self.current_round += 1
        for agent in self.active_agents:
            agent.start_interacting(agent in self.prison)
        self.set_time_for_interaction_session()
        self.start_interaction_timer()
        self.display(5, ["Interaction Mode"])
        self.display(6, [self.current_round])
        self.display(7, [self.resource_total])

    def end_timer(self):
        self.interaction_timer = None

    def stop_mining(self):
        self.in_mining_mode = False
        self.in_interaction_mode = False
        self.in_before_interaction_mode = True

    def elapsed_time(self):
        return time.time() - self.interaction_timer

    def stop_interactions(self):
        self.end_timer()
        self.in_interaction_mode = False
        self.in_before_interaction_mode = False
        self.in_mining_mode = True
        for agent in self.active_agents:
            agent.stop_interacting()

    def have_all_agents_stopped_interacting(self):
        all_stoped = True
        for agent in self.active_agents:
            if not agent.has_stopped_interaction:
                all_stoped = False
        return all_stoped

    def start_mining_mode(self):
        if self.should_test:
            self.agent_wealth_after_interaction_test = {agent: agent.wealth for agent in self.active_agents}
            self.agent_competency_after_interaction_test = { agent:agent.competency.copy() for agent in self.active_agents}

        self.in_interaction_mode = False
        self.in_before_interaction_mode = False
        self.in_mining_mode = True
        for agent in self.active_agents:
            agent.start_mining()
        self.display(5, ["Mining Mode"])

    def get_mined_amount(self, agent):
        n = agent.current_round - 1
        # a = n.o rounds for average miner to double the minimum mining
        a = 7
        mine = int(self.minimum_mine * (1 + 2*n*agent.competency.mining_skill/a))
        return mine

    def handle_mining(self):
        # Need to:
        # Give out money earnt to agents - update the current earnings

        improvement_percentage = 0.01

        agent_to_mined = {}
        for agent in self.active_agents:
            if agent not in self.prison:
                mine = self.get_mined_amount(agent)
                self.resource_total -= mine
                agent_to_mined[agent] = mine
                self.analysis.add_money_earnings(agent,mine,self.current_round)
                agent.increase_wealth(amount=mine,should_display=False,should_notify_environment=False,should_notify_all=False)

            # improve competency and personality for each agent
            e = agent.personality_template.dimension_percentage(agent.personality, 'X')
            c = agent.personality_template.dimension_percentage(agent.personality, 'C')

            m, a = max(agent.competency.mining_skill,0.001), max(agent.competency.appraisal_skill,0.001)
            improve_m = 1 + (improvement_percentage * c)
            improve_a = 1 + (improvement_percentage * e)
            m2 = min(m*improve_m,1)
            a2 = min(a*improve_a,1)

            self.analysis.add_competency_earnings(agent,m2-m,a2-a,self.current_round)

            agent.competency.update(m2,a2)

        sorted_x = sorted(agent_to_mined.items(), key=operator.itemgetter(1))

        self.display(2, [sorted_x, self.prison.copy()])
        self.display(4, [[agent.copy() for agent in self.active_agents]])
        self.display(7, [self.resource_total])

        agent_to_wealth = {}

        if self.should_test:
            self.agent_wealth_after_mining_test = { agent:agent.wealth for agent in self.active_agents}

        for agent in agent_to_mined:
            agent_to_wealth[agent] = agent.wealth
            self.notify_wealth_increase(agent=agent, amount=agent_to_mined[agent],should_notify_environment=True,should_notify_all=False,should_display=False)

        for agent in self.prison:
            agent_to_wealth[agent] = agent.wealth

        self.agent_earnings_after_mining = self.agent_earnings_current_round.copy()

        self.display(3, [agent_to_wealth])
        self.display(5, ["Adding Bonuses"])

        # Process all of the pending exchanges from interactions - update current earnings
        for interaction in self.interactions_to_promised_exchanges:
            exchanges = self.interactions_to_promised_exchanges[interaction]
            for exchange in exchanges:
                self.process_now(exchange)

        self.display(4, [[agent.copy() for agent in self.active_agents]])

        agent_to_wealth = {}
        # Redo appraisals
        for agent in self.active_agents:
            agent_to_wealth[agent] = agent.wealth
            for a in self.active_agents:
                if a != agent:
                    personality, competency = self.get_appraised_personality_and_competency(agent,a)
                    wealth = a.wealth
                    agent.update_personality(a,personality)
                    agent.update_competency(a, competency)
                    agent.update_wealth(a,wealth)
        self.display(3, [agent_to_wealth])

    def get_environment_ready_for_interactions(self):
        # Last earnings = Current earnings
        self.agent_earnings_last_round = self.agent_earnings_current_round.copy()

        self.prison = list(self.agent_to_prison_sentence.keys())

        to_free = []
        for agent in self.agent_to_prison_sentence:
            self.agent_to_prison_sentence[agent] -= 1
            if self.agent_to_prison_sentence[agent] < 1:
                to_free.append(agent)

        for agent in to_free:
            self.agent_to_prison_sentence.pop(agent)

        self.agent_earnings_current_round = {}
        for agent in self.active_agents:
            self.agent_earnings_current_round[agent] = 0

        #Reset all requested interactions
        self.confirmed_interactions = []
        self.interactions_to_promised_exchanges = {}

        # Reset all possible interactions
        for interaction in self.all_requested_interactions:
            interaction.reset()


        # for agent in self.active_agents:
        #     if agent in self.agent_to_possible_interactions:
        #         type_to_interactions = self.agent_to_possible_interactions[agent]
        #         for type in type_to_interactions:
        #             interactions = type_to_interactions[type]
        #             for interaction in interactions:
        #                 Interaction.reset(interaction)


    def notify_requested_interaction(self, interaction):
        self.lock_requested_interactions.acquire()
        if interaction not in self.all_requested_interactions:
            self.all_requested_interactions.append(interaction)
        self.lock_requested_interactions.release()


    def get_agent_earnings_list(self):

        self.access_agent_earning_current_lock.acquire()
        sorted_x = sorted(self.agent_earnings_current_round.items(), key=operator.itemgetter(1))
        self.access_agent_earning_current_lock.release()
        return sorted_x

    def stop_running(self):
        self.is_running = False
        for agent in self.active_agents:
            agent.stop_running()

    def run_test_on_test_variables(self):
        if self.should_test:
            print("Running tests at end of round")
            agent_wealth_at_end = {agent: agent.wealth for agent in self.active_agents}
            agent_competency_at_end = {agent: agent.competency.copy() for agent in self.active_agents}

            testInteraction = TestingInteractionSuite(
                [interaction for interaction in self.confirmed_interactions if interaction.is_success],
                self.agent_earnings_last_round.copy(),
                self.agent_wealth_before_interaction_test.copy(),
                self.agent_wealth_after_interaction_test.copy(),
                self.get_max_number_of_interactions_each_round(),
                self.agent_competency_before_interaction_test,
                self.agent_competency_after_interaction_test,
                Friendship, Mentorship, Help, Theft,
                self.minimum_mine, self.get_competency_increase_percentage())

            testRound = TestingRoundSuite(
                [interaction for interaction in self.confirmed_interactions if interaction.is_success],
                self.agent_earnings_after_mining.copy(),
                self.agent_wealth_after_mining_test, agent_wealth_at_end,
                self.agent_competency_after_interaction_test.copy(),
                agent_competency_at_end, Friendship, Mentorship, Help, Theft)

            suite = unittest.TestSuite()
            suite.addTests(testInteraction)
            suite.addTests(testRound)

            runner = unittest.TextTestRunner()
            runner.run(suite)

    def run(self):
        self.is_running = True
        start = time.time()

        while self.is_running:

            # Before starting interaction stage: add all agents
            if self.in_before_interaction_mode:
                self.include_all_agents()
                self.display(5, ["Including Agents"])
                self.display(4, [[agent.copy() for agent in self.active_agents]])
                self.display(3, [self.get_agent_to_wealth()])
                self.start_interactions()

            if self.in_interaction_mode and self.elapsed_time() > self.time_per_interaction_session:
                # interaction time has run out - need to get all agents to stop interacting
                self.stop_interactions()

            if self.in_mining_mode and self.have_all_agents_stopped_interacting():
                # now that all agents have stopped, we can switch to mining mode
                #x = "Interactions " + str([interaction for interaction in self.confirmed_interactions if interaction.is_success])
                #print(x)
                self.start_mining_mode()
                self.handle_mining()

                x = "Simulated round " + str(self.current_round)
                #print(x)

                self.run_test_on_test_variables()

                if (self.current_round + 1) > self.number_of_rounds:
                    self.stop_running()
                    print("Elapsed", (time.time() - start))
                    return self.display_requests, self.analysis

                self.get_environment_ready_for_interactions()
                self.stop_mining()

        self.stop_running()
        print("Elapsed", (time.time() - start))

        return self.display_requests, self.analysis


class Interaction:

    type = "Interaction"

    @classmethod
    def type(cls):
        return type(cls)

    def initiate_interaction(self):
        pass

    def __repr__(self):
        cool = self.type + "(" + str(self.proactive_agent) + "," + str(self.reactive_agent) + ")"
        return cool

    def __init__(self, proactive_agent, reactive_agent, single_role, request_bidirectional, requires_acceptance, should_always_notify_all, environment):
        if proactive_agent is None or reactive_agent is None:
            raise ValueError("Interaction given null agent value")
        self.proactive_agent = proactive_agent
        self.reactive_agent = reactive_agent
        self.environment = environment
        self.is_success = False
        self.is_accepted_or_rejected = False
        self.is_single_role = single_role
        self.should_always_notify_all = should_always_notify_all
        self.requires_acceptance = requires_acceptance
        self.is_request_bidirectional = request_bidirectional
        self.is_proactive_busy_during_initiation = True
        self.is_reactive_busy_during_initiation = self.requires_acceptance
        self.requested_agent = None
        self.access_request_lock = threading.Lock()

    def reset(self):
        self.requested_agent = None
        self.is_success = False
        self.is_accepted_or_rejected = False

    def is_busy_during_initiation(self, agent):
        return (self.reactive_agent == agent and self.is_reactive_busy_during_initiation) \
               or (self.proactive_agent == agent and self.is_proactive_busy_during_initiation)

    def contains(self,agent1, agent2):
        return self.is_present(agent1) and self.is_present(agent2)

    def is_present(self,agent):
        return self.proactive_agent==agent or self.reactive_agent==agent

    def get_requested_agent(self):
        return self.requested_agent

    def other_agent(self, agent):
        if not self.is_present(agent):
            return None
        return self.reactive_agent if self.proactive_agent == agent else self.proactive_agent

    def get_accepted_agent(self):
        if self.requested_agent is not None:
            return self.proactive_agent if self.reactive_agent==self.reactive_agent else self.reactive_agent
        return None

    def is_requestable_by(self, agent):
        is_proactive = agent == self.proactive_agent
        requestable = self.is_request_bidirectional or (not self.is_request_bidirectional and is_proactive)
        lim = round(self.environment.get_max_number_of_interactions_each_round())
        return requestable and self.can_happen() and agent.current_number_of_interactions < lim

    def other_agents(self, agent):
        if not self.is_present(agent):
            return [self.proactive_agent, self.reactive_agent]
        return [self.other_agent(agent)]

    def exceeded_interaction_limits(self):
        lim = round(self.environment.get_max_number_of_interactions_each_round())
        is_ok = not self.is_request_bidirectional and not self.requires_acceptance
        single = self.proactive_agent.current_number_of_interactions < lim
        both = single and self.reactive_agent.current_number_of_interactions < lim
        cond = (is_ok and not single) or (not is_ok and not both)
        return cond

    def request(self, agent):

        # Both agents need to be interacting
        if not(self.proactive_agent.is_interacting and self.reactive_agent.is_interacting):
            return None

        # Agent cant request it
        if not self.is_requestable_by(agent):
            return None

        self.access_request_lock.acquire()
        agent.acquire_interact_lock()

        # If already accepted, do nothing
        if self.is_accepted_or_rejected:
            agent.release_interact_lock()
            self.access_request_lock.release()
            return None

        ResourceMiningEnvironment.notify_requested_interaction(self.environment,self)
        # if doesn't need acceptance
        if not self.exceeded_interaction_limits() and not self.requires_acceptance:
            self.is_success = True
            self.is_accepted_or_rejected = True
            self.requested_agent = agent
            self.accept()
            agent.release_interact_lock()
            self.access_request_lock.release()
            return None

        # If a request already exists then accept
        if not self.exceeded_interaction_limits() and self.requested_agent is not None and not self.is_accepted_or_rejected:
            self.is_success = True
            self.accept()
            agent.release_interact_lock()
            self.access_request_lock.release()
            return None

        if not self.exceeded_interaction_limits():
            # If first request, then query other agent for response
            self.requested_agent = agent
            responding_agent = self.other_agent(agent)
            self.respond(responding_agent)
            agent.release_interact_lock()
            self.access_request_lock.release()
            return None

        # if any agent has gone over its interaction limit
        if self.exceeded_interaction_limits():
            self.is_success = False
            self.is_accepted_or_rejected = True
            if self.requested_agent is None:
                self.requested_agent = agent
            agent.release_interact_lock()
            self.access_request_lock.release()
            return None

    def can_agent_respond(self, agent):
        is_not_responded = not self.is_accepted_or_rejected
        is_requested = self.requested_agent is not None
        does_agent_belong = self.is_present(agent)
        is_not_requested_by_agent = (not self.requested_agent == agent)
        return does_agent_belong and is_not_responded and is_requested and is_not_requested_by_agent

    def respond(self, agent):
        if self.can_agent_respond(agent):
            response = agent.accept_interaction(self)
            if response is not None:
                response = response and not self.exceeded_interaction_limits()
                # this means interaction happened
                self.is_success = response
                self.accept()

    # NEED TO DO COPY
    def copy(self):
        pass

    def accept(self):
        self.is_accepted_or_rejected = True
        # need to tell environment that interaction happened before exchange goes through
        self.environment.notify_interaction(self)

        # needs to tell agents involved in interaction that it happened
        self.proactive_agent.increment_number_of_interactions(self)
        self.reactive_agent.increment_number_of_interactions(self)
        self.proactive_agent.interaction_happened(self)
        self.reactive_agent.interaction_happened(self)

        # Exchange only occurs if interactions are successful
        if self.is_success:
            self.exchange()
        if self.is_success and self.should_always_notify_all:
            self.notify_all()

    def get_is_success(self):
        return self.is_success

    def can_happen(self):
        return True

    def exchange(self):
        pass

    def notify_all(self):
        self.environment.notify_all_agents(self)

    def get_requires_acceptance(self):
        return self.get_requires_acceptance

    def get_is_bidirectional(self):
        return self.is_bidirectional

    def get_proactive_agent(self):
        return self.proactive_agent

    def get_reactive_agent(self):
        return self.reactive_agent


class Friendship(Interaction):
    type = "Friendship"
    is_single_role = True

    def __init__(self, proactive_agent, reactive_agent, environment):
        super(Friendship, self).__init__(proactive_agent, reactive_agent, True, True, True, True, environment)

    def exchange(self):
        friend_token = FriendshipToken(friend1=self.proactive_agent, friend2=self.reactive_agent,should_notify_all=False,should_display=False,should_notify_environment=True)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, friend_token, friend_token)
        ResourceMiningEnvironment.process_later(self.environment,self,exchange)

    def can_happen(self):
        # only condition is that other person accepts
        return True

    @classmethod
    def from_interaction(cls, friendship):
        a = cls(friendship.proactive_agent, friendship.reactive_agent, friendship.environment)
        a.is_accepted_or_rejected = friendship.is_accepted_or_rejected
        a.is_success = friendship.is_success
        a.requested_agent = friendship.requested_agent
        return a

    def copy(self):
        copy = Friendship.from_interaction(self)
        return copy


class Mentorship(Interaction):
    type = "Mentorship"
    is_single_role = False

    def __init__(self, proactive_agent, reactive_agent, environment):
        super(Mentorship, self).__init__(proactive_agent,reactive_agent,False,True,True, True, environment)

    def exchange(self):
        mentor_token = MentorshipToken(mentor=self.proactive_agent, mentee=self.reactive_agent,percentage=self.environment.get_competency_increase_percentage(), should_notify_all=False,should_display=False,should_notify_environment=True)
        mentor_token.add_competency(self.reactive_agent)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, mentor_token, mentor_token)
        ResourceMiningEnvironment.process_later(self.environment,self,exchange)

    @classmethod
    def from_interaction(cls, mentorship):
        a = cls(mentorship.proactive_agent,mentorship.reactive_agent,mentorship.environment)
        a.is_accepted_or_rejected = mentorship.is_accepted_or_rejected
        a.is_success = mentorship.is_success
        a.requested_agent = mentorship.requested_agent
        return a

    def copy(self):
        copy = Mentorship.from_interaction(self)
        return copy

    def can_happen(self):
        # only condition is that proactive_agent has a greater competency than reactive agent
        return self.proactive_agent.competency.contains_greater_competency_skill(self.reactive_agent.competency)


class Help(Interaction):
    type = "Help"
    is_single_role = False

    def reset(self):
        super(Help, self).reset()
        self.helping_funds = None

    def determine_helping_funds(self):
        self.helping_funds = self.environment.helping_amount(self.reactive_agent)

    def initiate_interaction(self):
        self.determine_helping_funds()

    @classmethod
    def from_interaction(cls, help):
        a = cls(help.proactive_agent, help.reactive_agent, help.environment)
        a.is_accepted_or_rejected = help.is_accepted_or_rejected
        a.is_success = help.is_success
        a.requested_agent = help.requested_agent
        a.helping_funds = help.helping_funds
        return a

    def copy(self):
        copy = Help.from_interaction(self)
        return copy

    def __init__(self, proactive_agent, reactive_agent, environment):
        super(Help, self).__init__(proactive_agent, reactive_agent,False,False,False, True, environment)
        self.helping_funds = None

    def get_helping_funds(self):
        return self.helping_funds

    def exchange(self):
        help_token = HelpToken(self.proactive_agent, self.reactive_agent, self.helping_funds, should_notify_all=True,should_display=True,should_notify_environment=True)
        exchange = Exchange(self.proactive_agent, self.reactive_agent,help_token, help_token)
        ResourceMiningEnvironment.process_now(self.environment,exchange)

    def can_happen(self):
        if self.helping_funds is None:
            self.determine_helping_funds()
        # proactive agent has enough funds to help
        return self.proactive_agent.wealth > self.helping_funds


class Theft(Interaction):
    type = "Theft"
    is_single_role = False

    def reset(self):
        super(Theft, self).reset()
        self.stolen_funds = None
        self.is_caught = None

    def initiate_interaction(self):
        self.determine_stolen_funds()

    def determine_stolen_funds(self):
        self.stolen_funds = self.environment.theft_amount(self.reactive_agent)

    @classmethod
    def from_interaction(cls, theft):
        a = cls(theft.proactive_agent, theft.reactive_agent, theft.environment)
        a.is_accepted_or_rejected = theft.is_accepted_or_rejected
        a.is_success = theft.is_success
        a.requested_agent = theft.requested_agent
        a.stolen_funds = theft.stolen_funds
        a.is_caught = theft.is_caught
        return a

    def copy(self):
        copy = Theft.from_interaction(self)
        return copy

    def __init__(self, proactive_agent, reactive_agent, environment):
        super(Theft, self).__init__(proactive_agent,reactive_agent,False,False,False, False, environment)
        self.stolen_funds = None
        self.is_caught = None
        self.reset()

    def accept(self):
        caught = self.determine_if_caught()
        super(Theft, self).accept()
        if caught:
            # Put thief in prison
            self.environment.add_to_prison_for_next_round(self.proactive_agent)
            self.proactive_agent.add_caught_theft_for_this_round(self)
            self.notify_all()

    def exchange(self):
        theft_token = TheftToken(self.proactive_agent, self.reactive_agent, self.stolen_funds, should_notify_all=True,should_display=True,should_notify_environment=True)
        exchange = Exchange(self.proactive_agent, self.reactive_agent, theft_token, theft_token)
        ResourceMiningEnvironment.process_now(self.environment, exchange)


    def get_stolen_funds(self):
        return self.stolen_funds

    def can_happen(self):
        # no condition
        return True

    def determine_if_caught(self):
        if self.is_caught is not None:
            return self.is_caught
        self.is_caught = random_boolean_variable(self.proactive_agent.caught_stealing())

        return self.is_caught


class Token:

    def remove_from(self, agent):
        pass

    def add_to(self, agent):
        pass

    def __init__(self, should_notify_all=True, should_notify_environment=True, should_display=True):
        self.should_notify_all = should_notify_all
        self.should_notify_environment = should_notify_environment
        self.shoud_display = should_display


class FriendshipToken(Token):

    def __repr__(self):
        x = "Add[" + self.friend1.name + "- Earns, " + self.friend2.name + "- Earns]"
        return "Friendship(" + x + ")"

    def remove_from(self, agent):
        return True

    def add_to(self, agent):
        if not (agent == self.friend1 or agent == self.friend2):
            return True

        friend = self.friend2 if agent == self.friend1 else self.friend1

        amount = ResourceMiningEnvironment.get_agent_earnings_after_mining(friend.environment, friend)
        amount *= 0.1
        amount = int(amount)

        if amount > 0:
            agent.environment.analysis.add_interaction_money_earnings(agent,agent==self.friend1, amount, Friendship, agent.environment.current_round)
            return ResourceToken.change(amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

        return True

    def __init__(self, friend1, friend2, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(FriendshipToken, self).__init__(should_notify_all,should_notify_environment,should_display)
        self.friend1 = friend1
        self.friend2 = friend2


class MentorshipToken(Token):

    def __repr__(self):
        x = "Add[" + self.mentee.name + "-(" + str(self.p) + "," + str(self.skill_list) + ")," + self.mentor.name + "-" + str(self.amount) + "]"
        r = "Remove[" + self.mentee.name + "-( EARNS )]"
        return "Mentorship(" + x + "," + r + ")"

    def add_competency(self, agent):
        # Only add competency to mentee
        if agent == self.mentee:
            CompetencyToken(self.p, self.skill_list,self.should_notify_all,self.should_notify_environment,self.shoud_display).add_to(agent)

    def add_to(self, agent):
        if agent == self.mentor:
            self.amount = self.get_mentee_earnings()
            if self.amount > 0:
                agent.environment.analysis.add_interaction_money_earnings(agent, True, self.amount,
                                                                          Mentorship, agent.environment.current_round)
                return ResourceToken.change(self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

    def get_mentee_earnings(self):
        if self.amount is None:
            self.amount = ResourceMiningEnvironment.get_agent_earnings_after_mining(self.mentee.environment, self.mentee)
            self.amount *= 0.25
            self.amount = int(self.amount)
        return self.amount

    def remove_from(self, agent):
        # Only remove from mentee
        if not self.mentee == agent:
            return True
        self.amount = self.get_mentee_earnings()
        if self.amount > 0:
            agent.environment.analysis.add_interaction_money_earnings(agent, False, -self.amount,
                                                                      Mentorship, agent.environment.current_round)
            return ResourceToken.change(-self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

        return True

    def __init__(self, mentor, mentee, percentage, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(MentorshipToken, self).__init__(should_notify_all,should_notify_environment,should_display)
        # will return list of skills better than mentee and difference for each of those skills
        self.skill_list, mp, ap = mentor.competency.better_skills(mentee.competency)
        self.mentor = mentor
        self.mentee = mentee
        self.amount = None
        if mp and ap is None:
            self.p = 0
        else:
            self.p = mp if (mp is not None and (ap is None or ap >= mp)) else ap
            # increasing by 15% of the smallest difference
            self.p *= percentage


class TheftToken(Token):

    def __repr__(self):
        x = "Add/Remove " + str(self.amount) + " to " + self.thief.name + "/" + self.victim.name
        return "Theft(" + x + ")"

    def add_to(self, agent):
        if agent == self.thief:
            if self.amount > 0:
                agent.environment.analysis.add_interaction_money_earnings(agent, True, self.amount,
                                                                          Theft, agent.environment.current_round)
                return ResourceToken.change(self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

    def remove_from(self, agent):
        if agent == self.victim:
            if self.amount > 0:
                agent.environment.analysis.add_interaction_money_earnings(agent, False, -self.amount,
                                                                          Theft, agent.environment.current_round)
                return ResourceToken.change(-self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

    def __init__(self, thief, victim, amount, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(TheftToken, self).__init__(should_notify_all=True, should_notify_environment=True, should_display=True)
        self.thief = thief
        self.victim = victim
        self.amount = amount


class HelpToken(Token):

    def __repr__(self):
        x = "Add/Remove " + str(self.amount) + " to " + self.helped.name + "/" + self.helper.name
        return "Help(" + x + ")"

    def add_to(self, agent):
        if agent == self.helped:
            if self.amount > 0:
                agent.environment.analysis.add_interaction_money_earnings(agent, False, self.amount,
                                                                          Help, agent.environment.current_round)
                return ResourceToken.change(self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

    def remove_from(self, agent):
        if agent == self.helper:
            if self.amount > 0:
                agent.environment.analysis.add_interaction_money_earnings(agent, True, -self.amount,
                                                                          Help, agent.environment.current_round)
                return ResourceToken.change(-self.amount, agent, self.should_notify_all, self.should_notify_environment, self.shoud_display)

    def __init__(self, helper, helped, amount, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(HelpToken, self).__init__(should_notify_all=True, should_notify_environment=True, should_display=True)
        self.helped = helped
        self.helper = helper
        self.amount = amount


class ResourceToken(Token):

    def remove_from(self, agent):
        amount = -1 * self.amount
        return ResourceToken.change(amount, agent)

    def add_to(self, agent):
        amount = self.amount
        return ResourceToken.change(amount,agent)

    @staticmethod
    def change(amount, agent, should_notify_all=True, should_notify_environment=True, should_display=True):
        agent.increase_wealth(amount,should_notify_all=should_notify_all,should_notify_environment=should_notify_environment,should_display=should_display)
        return True

    def __init__(self, amount, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(ResourceToken, self).__init__(should_notify_all=True, should_notify_environment=True, should_display=True)
        self.amount = amount


class CompetencyToken(Token):

    def __repr__(self):
        x = "Remove/Add " + str(self.percentage) + " " + str(self.skill_list)
        return "Competency(" + x + ')'

    def remove_from(self, agent):
        return self.change_competency(self.amount, agent)

    def change_competency(self, p, agent):
        for s in self.skill_list:
            if s == "appraisal":
                agent.environment.analysis.add_interaction_comp_earnings(agent, False, 0, p,
                                                                         Mentorship, agent.environment.current_round)
                agent.competency.increase_appraisal_skills(p)
            else:
                agent.environment.analysis.add_interaction_comp_earnings(agent, False, p, 0,
                                                                         Mentorship, agent.environment.current_round)
                agent.competency.increase_mining_skills(p)
        return True

    def add_to(self, agent):
        return self.change_competency(self.amount,agent)

    def __init__(self, amount, skill_list, should_notify_all=True, should_notify_environment=True, should_display=True):
        super(CompetencyToken, self).__init__(should_notify_all=True, should_notify_environment=True, should_display=True)
        self.amount = amount
        self.skill_list = skill_list


class Exchange:

    def process(self):
        # remove lost_token from from_agent
        self.lost_token.remove_from(self.from_agent)
        self.lost_token.remove_from(self.to_agent)

        # add gain_token to to_token
        self.gained_token.add_to(self.to_agent)
        self.gained_token.add_to(self.from_agent)

    def __repr__(self):
        return "Exchange: " + "[" + str(self.from_agent) + "," + str(self.to_agent) + "],[" + str(self.lost_token) + " " + str(self.gained_token) + "]"

    # during exchange: from_agent loses lost_token and to_agent gains gained_token
    def __init__(self, from_agent, to_agent, lost_token, gained_token):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.lost_token = lost_token
        self.gained_token = gained_token























