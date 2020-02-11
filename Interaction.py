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


class Mentorship(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, True, True, environment)

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass


class Help(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, False, False, environment)
        self.helping_funds = 100

    def request(self, agent):
        pass

    def get_helping_funds(self):
        return self.helping_funds

    def exchange(self):
        pass

    def notify_all(self):
        pass



class Theft(Interaction):

    def __init__(self, proactive_agent, reactive_agent, environment):
        Interaction.__init__(self,proactive_agent, reactive_agent, False, False, environment)
        self.stolen_funds = 100
        self.is_caught = False

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def get_stolen_funds(self):
        return self.stolen_funds

