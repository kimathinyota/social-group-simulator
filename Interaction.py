from abc import ABC, abstractmethod


class Interaction(ABC):

    def __init__(self, proactive_agent, reactive_agent, is_bidirectional, requires_acceptance, environment):
        self.proactive_agent = proactive_agent
        self.reactive_agent = reactive_agent
        self.environment = environment
        self.is_success = False
        self.is_bidirectional = is_bidirectional
        self.requires_acceptance = requires_acceptance
        self.requested_agent = None

    def reset(self):
        self.requested_agent = None
        self.is_success = False

    def contains(self,agent1, agent2):
        return (self.proactive_agent==agent1 or self.requested_agent==agent1) and (self.proactive_agent==agent2 or self.requested_agent==agent2)

    def get_requested_agent(self):
        return self.requested_agent

    def get_accepted_agent(self):
        if self.requested_agent is not None:
            return self.proactive_agent if self.reactive_agent==self.reactive_agent else self.reactive_agent
        return None

    def request(self, agent):
        pass

    def get_is_success(self):
        return self.is_success

    def can_happen(self):
        pass

    def probability_of_occurring(self):
        pass



    def exchange(self):
        pass

    def notify_all(self):
        pass

    def get_requires_acceptance(self):
        return self.get_requires_acceptance

    def get_is_bidirectional(self):
        return self.is_bidirectional

    def get_proactive_agent(self):
        return self.proactive_agent

    def get_reactive_agent(self):
        return self.reactive_agent


class Friendship(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent,reactive_agent,True, True, environment)

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def can_happen(self):
        # only condition is that other person accepts
        return True


class Mentorship(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, False, True, environment)

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def can_happen(self):
        # only condition is that proactive_agent has a greater competency than reactive agent
        return self.proactive_agent.competency.contains_greater_competency_skill(self.reactive_agent.competency)


class Help(Interaction):

    def reset(self):
        super(Theft, self).reset()
        self.helping_funds = self.environment.helping_funds

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, False, False, environment)
        self.helping_funds = self.environment.helping_funds

    def request(self, agent):
        pass

    def get_helping_funds(self):
        return self.helping_funds

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def can_happen(self):
        # proactive agent has enough funds to help
        return self.proactive_agent.wealth > self.helping_funds


class Theft(Interaction):

    def reset(self):
        super(Theft, self).reset()
        self.stolen_funds = self.environment.theft_amount(self.reactive_agent)
        self.is_caught = None

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, False, False, environment)
        self.stolen_funds = 0
        self.is_caught = None
        self.reset()

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def get_stolen_funds(self):
        return self.stolen_funds

    def can_happen(self):
        # no condition
        return True

    def determine_if_caught(self):
        if self.is_caught is not None:
            return self.is_caught
        # More you steal, less likely you are to be caught

