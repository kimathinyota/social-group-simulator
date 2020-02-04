from src.ServiceGUI import *
import threading

class SocialGroup:

    # constructor for SocialGroup
    def __init__(self, gui_server, gui_server_lock):

        # store all of the agents within this social group
        self.agents = []

        # store list of personality facets valued or unvalued by the group
        self.wanted_group_values = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
        self.unwanted_group_values = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
        # store list of personality facets the social group considers to be 'dominant'
        self.dominant_values = ['E1', 'E2', 'E3', 'O5', 'C1', 'C4', 'C5']
        # store list of personality facets the social group considers to be 'non-dominant'
        self.non_dominant_values = ['N1', 'N3', 'N6']

        # maps agent to string identifier for interaction
        self.interactions = {}

        # maps interaction identifier -> (agent, action) map
        self.interaction_actions = {}

        self.interaction_lock = threading.Lock()

        self.number_of_interactions = 0

        # for sending display requests to the GUI
        self.gui_server = gui_server
        # lock to avoid race conditions within gui_server
        self.gui_server_lock = gui_server_lock

    def set_group_values(self, wanted_group_values, unwanted_group_values):
        self.wanted_group_values = wanted_group_values
        self.unwanted_group_values = unwanted_group_values

    def set_dominant_values(self, dominant_values, non_dominant_values):
        self.dominant_values = dominant_values
        self.non_dominant_values = non_dominant_values

    def get_action_during_interaction(self, agent, other_agent):

        self.interaction_lock.acquire()

        # check if agent is currently in an interaction
        if agent not in self.interactions or self.interactions[agent] is None:
            return None
        name = self.interactions[agent]

        result = self.interaction_actions[name][other_agent]

        self.interaction_lock.release()

        return result

    def get_appraisal_during_interaction(self, agent):
        # print("Get appraisal for agent: " + str(agent))
        if agent not in self.interactions or self.interactions[agent] is None:
            return None

        name = self.interactions[agent]

        completed_num = 0
        total_appraisal = 0

        interactions_actions = self.interaction_actions[name]

        for a in self.interaction_actions[name]:
            # check if this agent has completed the interaction
            if a != agent and (self.interactions[a] is None or self.interactions[a] != name):
                total_appraisal += a.appraisal(agent)
                completed_num += 1

        if completed_num >= max(0.5*len(self.interaction_actions[name])-1,1):
            return total_appraisal/completed_num

        return None

    # find the total status of the input agent
    def status_agent(self, agent):
        tot = 0
        # each agent will have assigned a status for this input agent
        # below will total up all of these statuses
        k = 0
        for a in self.agents:
            if a != agent:
                s = a.status_agent(agent)
                if s is not None:
                    tot += s
                    k += 1

        if k == 0:
            return 0

        return tot/k

    # will return a map of agent to status of all agents
    def get_agent_statuses(self):
        agent_to_status = {}
        for a in self.agents:
            agent_to_status[a] = round(self.status_agent(a), 3)
        return agent_to_status

    def add_agent(self, agent):

        agent.set_social_group(self)
        self.agents.append(agent)

        # Handling GUI display
        # copy is used to avoid strange concurrency errors
        agents = self.agents.copy()
        agents.sort(key=operator.attrgetter("name"))

        # need to acquire lock in order to send request to GUI server
        self.gui_server_lock.acquire()
        # request 4: Draw agents and their corresponding personalities in the personality section
        self.gui_server.service_request(4, [agents])
        # request 3: Draw agents and their corresponding statuses in order in the status hierarchy part
        self.gui_server.service_request(3, [self.get_agent_statuses()])
        self.gui_server_lock.release()

    # send request, in place of input agent, for interactions
    def find_agent(self, agent):
        # haven't simulated reason for interactions yet so ...
        # each agent will have a random chance of first meeting each other agent
        random.shuffle(self.agents)
        for a in self.agents:
            # agent can't interact with itself
            # each agent has a choice on whether to accept each interaction request
            if a != agent and a.interact_accept(agent):
                return a
        return None

    def find_and_start_interactions(self, agent):
        found_agent = self.find_agent(agent)
        if found_agent is None:
            return False
        self.start_interactions(agent,found_agent)
        return True

    # facilitates the start of an interaction when called by an agent
    def start_interactions(self, agent_one, agent_two):
        #print("Start interaction between: " + str(agent_one) + " and " + str(agent_two))

        # only one agent calls this method
        # we need to start these interactions at the same time (roughly)

        self.number_of_interactions += 1

        print("Number of interactions: " + str(self.number_of_interactions) )

        self.interaction_lock.acquire()
        agent_one.start_interaction(agent_two)
        agent_two.start_interaction(agent_one)

        # need a somewhat unique name for interaction - using a random number doesn't ensure uniqueness but
        # there's a high probability that it will be

        name = agent_one.name + "-" + agent_two.name + "-" + str(random.randrange(1,9000))


        # need to assign this interaction to the agents involved
        self.interactions[agent_one] = name
        self.interactions[agent_two] = name
        self.interaction_actions[name] = { agent_one:agent_one.action(), agent_two:agent_two.action() }
        self.interaction_lock.release()

        # GUI: Draw this interaction
        self.gui_server_lock.acquire()
        self.gui_server.service_request(1, [self.agents.copy(), name])
        self.gui_server_lock.release()



    def should_delete_interaction(self, name):
        if name not in self.interaction_actions:
            return False
        agents_map = self.interaction_actions[name]
        should_delete = True
        for a in agents_map:
            if self.interactions[a] == name:
                should_delete = False

        return should_delete

    def delete_interaction(self, name):
        if self.should_delete_interaction(name):
            self.interaction_actions.pop(name)

    # called by agent for facilitating the end of the current interaction this agent is involved in
    def end_interaction(self, agent):
        #print("Ending interaction with " + str(agent))
        agent.end_interaction()

        # agent isn't currently interacting
        if agent not in self.interactions or self.interactions[agent] is None:
            return

        self.interaction_lock.acquire()
        # get name of current interaction this agent is involved in
        name = self.interactions[agent]

        self.interactions[agent] = None

        self.delete_interaction(name)

        agents = self.agents.copy()

        self.interaction_lock.release()

        agents.sort(key=operator.attrgetter("name"))

        # GUI: Un-draw this interaction
        self.gui_server_lock.acquire()
        self.gui_server.service_request(2, [name])
        self.gui_server.service_request(3, [self.get_agent_statuses()])
        self.gui_server.service_request(4, [agents])
        self.gui_server_lock.release()






















