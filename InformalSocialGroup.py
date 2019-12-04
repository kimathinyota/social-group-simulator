from src.ServiceGUI import *

class SocialGroup:
    def __init__(self, gui_server, gui_server_lock):
        self.agents = []
        self.wanted_group_values = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
        self.unwanted_group_values = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
        self.dominant_values = ['E1', 'E2', 'E3', 'O5', 'C1', 'C4', 'C5']
        self.non_dominant_values = ['N1', 'N3', 'N6']
        self.gui_server = gui_server
        self.interactions = {}
        self.gui_server_lock = gui_server_lock

    def set_group_values(self, wanted_group_values, unwanted_group_values):
        self.wanted_group_values = wanted_group_values
        self.unwanted_group_values = unwanted_group_values

    def set_dominant_values(self, dominant_values, non_dominant_values):
        self.dominant_values = dominant_values
        self.non_dominant_values = non_dominant_values

    def status_agent(self, agent):
        tot = 0
        for a in self.agents:
            if a != agent:
                tot += a.status_agent(agent)
        return tot

    def print_agent_statuses(self):
        line = ''
        for a in self.agents:
            line += a.name + " - " + str(self.status_agent(a)) + ', '

        print(line)

    def get_agent_statuses(self):
        agentToStatus = {}
        for a in self.agents:
            agentToStatus[a] = round(self.status_agent(a), 3)
        return agentToStatus

    def add_agent(self, agent):
        agent.set_social_group(self)
        self.agents.append(agent)
        agents = self.agents.copy()

        self.gui_server_lock.acquire()
        self.gui_server.service_request(4, [agents])
        self.gui_server.service_request(3, [self.get_agent_statuses()])
        self.gui_server_lock.release()

    def find_agent(self, agent):
        random.shuffle(self.agents)
        for a in self.agents:
            if a != agent and a.interact_accept(agent):
                #print("For agent: " + agent.name + " we found agent " + a.name)
                return a
        return None

    def start_interactions(self, agent_one, agent_two):
        agent_one.start_interaction(agent_two)
        agent_two.start_interaction(agent_one)
        name = agent_one.name + "-" + agent_two.name + "-" + str(random.randrange(1,9000))
        self.interactions[agent_one] = name
        self.interactions[agent_two] = name
        self.gui_server_lock.acquire()
        self.gui_server.service_request(1, [self.agents, name])
        self.gui_server_lock.release()



        #print("Started interaction between: " + agent_one.name + " " + agent_two.name)

    def end_interaction(self, agent):
        agent.end_interaction()
        #print("Ended interaction for: " + agent.name)
        if agent not in self.interactions:
            pass
        name = self.interactions[agent]
        self.gui_server_lock.acquire()
        self.gui_server.service_request(2, [name])
        self.gui_server.service_request(3, [self.get_agent_statuses()])
        self.gui_server_lock.release()




















