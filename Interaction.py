from abc import ABC, abstractmethod


class Interaction(ABC):

    def __init__(self, proactive_agent, reactive_agent, environment):
        self.proactive_agent = proactive_agent
        self.reactive_agent = reactive_agent
        self.environment = environment

    def request(self, agent):
        pass

    def exchange(self):
        pass

    def notify_all(self):
        pass

    def is_bidirectional(self):
        pass

    def proactive_agent(self):
        return self.proactive_agent

    def reactive_agent(self):
        return self.reactive_agent()


