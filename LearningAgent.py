from src.Agent import *


class LearningAgent(Agent):

    def __init__(self, name, competency, personality):
        super(LearningAgent, self).__init__(name, competency, personality)

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
        if isinstance(interaction, Friendship):
            accept = self.accept_friend_probability(other)
            return random_boolean_variable(accept)
        elif isinstance(interaction, Mentorship):
            e = self.environment.estimated_earnings(self)
            accept = self.accept_mentor_probability(other, e) if self == interaction.reactive_agent \
                else self.accept_mentee_probability(other)
            return random_boolean_variable(accept)

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
                            choice = random.choices(population=interactions, weights=weights, k=1)[0]
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
