from src.Agent import *
import functools


class LearningAgent(Agent):

    type = "Learning"

    # Q matrix:[competency state: 10][round state: 10][agent variables: 8][agent level: 5][interaction choice: 9]

    def __init__(self, name, competency, personality, generation_id=None, is_training=True, long_term_round_count=5, q_array_sizes=[10, 10, 8, 5, 9], Q=None, discount=0.85, gamma=0.9, alpha=0.4, interaction_threshold = 0.2):
        super(LearningAgent, self).__init__(name, competency, personality, generation_id)
        # q_array_sizes: [#competency state][#round states][#agent variables][#agent level][#interaction choice]
        self.q_array_sizes = q_array_sizes
        # initalise Q with zeros
        self.Q = Q
        if self.Q is None:
            self.Q = np.array(np.zeros(q_array_sizes))
        # this agent's earnings each round
        self.my_earnings_each_round = []
        self.my_mining_earnings_each_round = []
        self.is_training = is_training
        self.long_term_round_count = long_term_round_count
        self.q_lock = threading.Lock()
        # each action has a long-term effect that'll need to be rewarded/punished so updating q-values will be delayed
        self.delayed_q_entries = []
        self.discount = discount
        self.gamma = gamma
        self.alpha = alpha
        self.action_probabilities = []
        self.var_map = {"CM": 0, "CA": 1, "PH": 2, "PE": 3, "PX": 4, "PA": 5, "PC": 6, "PO": 7}
        self.interaction_threshold = interaction_threshold

    @staticmethod
    def position(value, number_levels, s=0, e=1):
        pos = math.floor(value/(e-s) * number_levels)
        return min(pos, number_levels - 1)

    def reward(self, interaction, start, finish):
        # start = round where interaction took place
        # finish = round where impact of interaction is said to have ended
        # reward = money_earnt(interaction) + SUM_(start->finish){discount^(round-start)*money_earnt(round)}
        interaction_to_earned = self.interaction_to_earned[start] if start in self.interaction_to_earned else {}
        # calculate total absolute earned in start
        # using this method instead of my_earnings_each_round: concerned with absolute ratios ...
        # ... so -7, -2, 1: (-7/10, -2/10, 1/10) not (7/8, 2/8, -1/8)
        # important if sum is 0: -6, 4, 2: (-6/12, 4,12, 2, 12)
        total = functools.reduce(lambda x, y: abs(x)+y, interaction_to_earned.values()) if len(interaction_to_earned) > 0 else 0
        total += self.my_mining_earnings_each_round[start]
        iid = str(interaction.id)
        # iearn = amount earned by interaction
        iearn = interaction_to_earned[iid] if iid in interaction_to_earned else 0
        # print("Earned: ", iearn, iid in interaction_to_earned, total)
        iearn /= (total if total > 0 else 1)

        # end <= len(my_earnings_each_round)
        end = min(start + finish+1, len(self.my_earnings_each_round))
        earnings = self.my_earnings_each_round[start:end]

        # Average earning
        avg = (functools.reduce(lambda x, y: abs(x) + y, earnings)) / len(earnings)

        # Normalise earnings
        r = math.pow(10, int(math.log(avg, 10) + 1))
        x = [i/r for i in earnings]

        total = 0
        # Find cumalitive total of discounted normalised earnings
        for i in range(len(x)):
            total += x[i]*math.pow(self.discount, i+1)

        m = 0
        for i in range(1, len(earnings) + 1):
            m += math.pow(self.discount, i)
        score = iearn + total/m
        # print([interaction, interaction.is_caught if isinstance(interaction,Theft) else None, [score, iearn, total], earnings])
        return score

    def get_other_state(self, agent):
        # 8 variables: [Mining, Appraisal, H, E, X, A, C, O]
        level = self.q_array_sizes[3]
        m, a = agent.competency.mining_skill, agent.competency.appraisal_skill
        state = [self.position(m, level), self.position(a, level)]
        personality = ['H', 'E', 'X', 'A', 'C', 'O']
        for p in personality:
            percent = agent.personality_template.dimension_percentage(agent.personality, p)
            state.append(self.position(percent, level))
        return tuple(state)

    def get_round(self, round):
        return self.position(round, self.q_array_sizes[1], 0, 40)

    def get_my_comp_state(self, competency):
        # first self.Q[0:max/2][...] --> will be for mining
        # second self.Q[max/2:max][...] --> will be for appraisal
        levels = self.q_array_sizes[0]/2
        m, a = competency.mining_skill, competency.appraisal_skill
        pm, pa = self.position(m, levels), self.position(a, levels)
        pa += levels
        return int(pm), int(pa)

    def interaction_happened(self, interaction):
        super(LearningAgent, self).interaction_happened(interaction)
        # x = "Hap: " + str(interaction) + " " + str(interaction.id)
        # print(x)
        if not (interaction.requested_agent != self and not interaction.requires_acceptance):
            # in this situation, this agent has no control and so nothing can be learnt from it
            # deep copy used: we want a snapshot of the interaction at the time (all data preserved)
            icopy = interaction.copy(is_deep=True)
            start = self.current_round - 1
            self.q_lock.acquire()
            # entry format (Interaction, start, finish)
            entry = (icopy, start, start + self.long_term_round_count)
            self.delayed_q_entries.append(entry)
            self.q_lock.release()

    @staticmethod
    def get_action_position(interaction_type, is_success, is_proactive):
        if is_success and interaction_type == Friendship:
            return 0
        if not is_success and interaction_type == Friendship:
            return 1
        if is_success and is_proactive and interaction_type == Mentorship:
            return 2
        if not is_success and is_proactive and interaction_type == Mentorship:
            return 3
        if is_success and not is_proactive and interaction_type == Mentorship:
            return 4
        if not is_success and not is_proactive and interaction_type == Mentorship:
            return 5
        if is_success and is_proactive and interaction_type == Theft:
            return 6
        if is_success and is_proactive and interaction_type == Help:
            return 7

    @staticmethod
    def get_action_accept_positions(interaction_type, is_success, is_proactive):
        p = LearningAgent.get_action_position(interaction_type, is_success, is_proactive)
        k = None
        if p < 2:
            k = 1 if p == 0 else 0
        elif p < 4:
            k = 2 if p == 3 else 3
        elif p < 6:
            k = 4 if p == 5 else 4
        return p, k

    def learn_from_interaction(self, interaction, start, finish):
        # Format: Type IsProactive IsPick Probability Agent
        # print("Learn", interaction, start, finish)
        # print("Keys", list(self.action_probabilities[start].keys()))
        # print("Bro", self.action_probabilities)
        probabilities = self.action_probabilities[start][str(interaction.id)]

        is_proactive = (interaction.proactive_agent.generation_id == self.generation_id)
        action = self.get_action_position(type(interaction), interaction.is_success, is_proactive)
        reward = self.reward(interaction, start, finish)

        other = interaction.reactive_agent if is_proactive else interaction.proactive_agent
        me = interaction.proactive_agent if is_proactive else interaction.reactive_agent

        comp_states = self.get_my_comp_state(me.competency)
        # print("-------------------------------------------------------------------------------------------------")
        for c in comp_states:
            for avar in self.var_map.keys():
                sum = 0
                for p in probabilities:
                    t, proactive, pick, prob, agent = p
                    if prob > 0:
                        a = self.get_action_position(t, pick, proactive)
                        q_value = self.get_q_value(c, agent, start, avar, a)
                        sum += prob*q_value
                sum *= self.gamma

                positions = self.get_q_positions(c, other, start, avar, action)
                comp_state, round_pos, var_pos, var_level, action = positions

                q = self.Q[comp_state][round_pos][var_pos][var_level][action]
                td = (reward + sum) - q
                # x = str([interaction, start, finish]) + " " + str(interaction.is_success) + " Q " + str(comp_state) + " " + str(round_pos) + " " + str(var_pos) + " " + str(var_level) + " " + str(action) \
                #     + " " + str([reward, q, td])
                # print(x)
                self.Q[comp_state][round_pos][var_pos][var_level][action] += self.alpha*td

    def get_q_positions(self, comp_state, agent, round, agent_variable, action):
        if agent_variable not in self.var_map.keys():
            return None
        states = self.get_other_state(agent)
        var_pos = self.var_map[agent_variable]
        var_level = states[var_pos]
        round_pos = self.get_round(round)
        return comp_state, round_pos, var_pos, var_level, action

    def get_q_value(self, comp_state, agent, round, agent_variable, action):
        positions = self.get_q_positions(comp_state, agent, round, agent_variable, action)
        if positions is None:
            return None
        comp_state, round_pos, var_pos, var_level, action = positions
        # [5, 10, 8, 5, 9]
        return self.Q[comp_state][round_pos][var_pos][var_level][action]

    def start_interacting(self, in_prison=False):
        self.action_probabilities.append({})
        super(LearningAgent, self).start_interacting(in_prison)

    def learn(self):
        remove = []
        for i in range(len(self.delayed_q_entries)):
            entry = self.delayed_q_entries[i]
            interaction, start, finish = entry
            if self.current_round < finish and (self.current_round-1) < self.environment.number_of_rounds:
                # hasn't reached that point yet - delayed_q_entries is in chronological order
                for r in remove:
                    self.delayed_q_entries.remove(r)
                return None
            self.learn_from_interaction(interaction, start, finish)
            remove.append(entry)

    def stop_mining(self):
        self.my_earnings_each_round.append(self.environment.get_agent_earnings_this_round(self))
        self.my_mining_earnings_each_round.append(self.environment.get_agent_earnings_after_mining(self))
        if self.is_training:
            self.learn()

    def accept_interaction_from_training(self, interaction):
        # Find max success interaction for interaction given current state
        is_proactive = (interaction.proactive_agent.generation_id == self.generation_id)
        accept_score = 0
        naccept_score = 0
        action, naction = self.get_action_accept_positions(type(interaction), interaction.is_success, is_proactive)
        for c in self.get_my_comp_state(self.competency):
            for avar in self.var_map.keys():
                comp_state, round_pos, var_pos, var_level, action = self.get_q_positions(c, interaction.other_agent(self),
                                                                                         self.current_round, avar, action)
                accept_score += self.Q[comp_state][round_pos][var_pos][var_level][action]
                naccept_score += self.Q[comp_state][round_pos][var_pos][var_level][naction]

        if accept_score == 0 and naccept_score == 0:
            return None

        p = int(accept_score) + int(naccept_score)
        a, b = int(accept_score)/p, int(naccept_score)/p

        choice = random.choices(population=[accept_score > 0, naccept_score < 0], weights=[a,b], k=1)[0]

        return choice

    def probable_action(self, agent, allowed_actions=[0,2,4,6,7], probabilities=[0.5, 0.5, 0.5, 0.5, 0.5]):
        comp_state = self.get_my_comp_state(self.competency)
        #itotals = [F,MP,MR,T,H]
        itotals = { u:0 for u in allowed_actions}
        for c in comp_state:
            for avar in self.var_map:
                comp_state, round_pos, var_pos, var_level, action = self.get_q_positions(c, agent, self.current_round,
                                                                                         avar, None)
                for k in itotals:
                    itotals[k] += self.Q[comp_state][round_pos][var_pos][var_level][k]

        actions = list(itotals.keys())
        # Using negatives: Ignore them
        #TODO: How can negatives be used in choosing best action?
        weights = [itotals[k] if itotals[k] > 0 else 0 for k in actions]
        sum = functools.reduce(lambda x, y: abs(x) + y, weights)
        if sum != 0:
            actual_sum = functools.reduce(lambda x, y: x/sum + y, weights)
        if sum == 0:
            weights = probabilities
            sum = functools.reduce(lambda x, y: abs(x) + y, weights)
            actual_sum = sum
        weights = [x/sum for x in weights]
        choice = random.choices(population=actions, weights=weights, k=1)[0]
        return choice, (actual_sum/len(weights))

    def request_action(self, agents, possible_interactions, weights):
        actionsm = {agent: [] for agent in agents}
        interactions = {agent: {} for agent in agents}
        i_to_weight = {possible_interactions[i]:weights[i] for i in range(len(possible_interactions))}
        for interaction in possible_interactions:
            action = self.get_action_position(type(interaction), True, interaction.proactive_agent == self)
            other = interaction.other_agent(self)
            actionsm[other].append(action)
            interactions[other][action] = interaction

        total_prob = 0
        pop = []
        weights = []

        for agent in agents:
            actions = actionsm[agent]
            if len(actions) > 0:
                default_probs = [i_to_weight[interactions[agent][action]] for action in actions]
                # print(actions)
                action, score = self.probable_action(agent, actions, default_probs)
                total_prob += score
                pop.append((agent, action))
                weights.append(score)

        avg = total_prob / len(agents)
        weights = [w/total_prob for w in weights]
        if avg < self.interaction_threshold:
            return None
        choice = random.choices(population=pop, weights=weights, k=1)[0]
        agent, action = choice
        return interactions[agent][action]

    def accept_interaction(self, interaction):
        if not interaction.is_present(self):
            return None

        if not self.is_training:
            # use data learnt from training
            should_accept = self.accept_interaction_from_training(interaction)
            if should_accept is None:
                return super(LearningAgent, self).accept_interaction(interaction)
            return should_accept

        if self.is_busy:
            self.access_pending_interactions_lock.acquire()
            self.pending_interaction_requests.append(interaction)
            self.access_pending_interactions_lock.release()
            return None

        # Does agent even wan't to interact ?
        interact_prob = self.interact_probability()
        should_interact = random_boolean_variable(interact_prob)
        if not should_interact:
            return None

        # Agent not busy and so can deal with request
        other = interaction.other_agent(self)

        # Format: Type IsProactive IsPick Probability Agent
        # only Friendship and Mentorship interactions require acceptance
        if isinstance(interaction, Friendship):
            accept = self.accept_friend_probability(other)
            pick = accept * interact_prob
            # print(self.action_probabilities, self.current_round-1)
            # x = "Accept: " + str(interaction) + " " + str(interaction.id)
            # print(x)
            self.action_probabilities[self.current_round-1][str(interaction.id)] = [(Friendship, True, True, pick,
                                                                                     other.copy()),
                                                                                    (Friendship, True, False, (1-pick),
                                                                                     other.copy())]
            return random_boolean_variable(accept)
        elif isinstance(interaction, Mentorship):
            # x = "Accept: " + str(interaction) + " " + str(interaction.id)
            # print(x)
            e = self.environment.estimated_earnings(self)
            is_reactive = self == interaction.reactive_agent
            accept = self.accept_mentor_probability(other, e) if is_reactive else self.accept_mentee_probability(other)
            is_p = not is_reactive
            pick = accept * interact_prob
            self.action_probabilities[self.current_round-1][str(interaction.id)] = [(Mentorship, is_p, True, pick,
                                                                                     other.copy()),
                                                                                    (Mentorship, is_p, False, (1-pick),
                                                                                     other.copy())]
            return random_boolean_variable(accept)

    @staticmethod
    def random(name):
        return LearningAgent(name, Competency.random(), HexacoPersonality().random_personality())

    def stop_running(self):
        super(LearningAgent, self).stop_running()
        # x = "Stopped running: " + str(self) + " " + str(self.competency) + " " + str(self.personality)
        # print(x)

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
                    interactions, weights, average_probability = self.environment.get_requestable_interactions(self)

                    if interactions is not None:
                        if len(interactions) > 0:
                            if self.is_training:
                                x = "Interactions: " + str(interactions)
                                #print(x)
                                choice = random.choices(population=interactions, weights=weights, k=1)[0]
                                # x = "Choice: " + str(choice) + " " + str(choice.id)
                                # print(x)
                                choice.request(self)
                                if str(choice.id) not in self.action_probabilities[self.current_round-1]:
                                    probs = [(type(interactions[i]), interactions[i].proactive_agent == self, True,
                                              weights[i], interactions[i].other_agent(self).copy())
                                             for i in range(len(interactions))]
                                    self.action_probabilities[self.current_round-1][str(choice.id)] = probs
                            else:
                                interaction = self.request_action(self.get_agents(), interactions, weights)
                                if interaction is not None:
                                    interaction.request(self)

                # Agent will now respond to up to two received interactions
                self.is_busy = True
                respond_to = []
                self.access_pending_interactions_lock.acquire()
                for i in range(min(2, len(self.pending_interaction_requests))):
                    respond_to.append(self.pending_interaction_requests[i])
                self.access_pending_interactions_lock.release()

                for interaction in respond_to:
                    interaction.respond(self)
            self.has_stopped_interaction = not self.is_interacting

