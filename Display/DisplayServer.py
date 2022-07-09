from src.Display.SocialGroupGUI import *
import threading


class ServiceGUI:
    def __init__(self):
        self.lock = threading.Lock()
        self.requestsList = []
        self.gui = SocialGroupGUI(1300, 700)

    @staticmethod
    def construct_request(request_code, args):
        request = {'type': request_code, 'args': args}
        return request

    def service_request(self, request_code, args):
        self.lock.acquire()
        self.requestsList.append(self.construct_request(request_code,args))
        self.lock.release()

    def service_requests(self, requests):
        requests = []
        for request in requests:
            code, args = request
            requests.append(self.construct_request(code,args))
        self.lock.acquire()
        self.requestsList += requests
        self.lock.release()

    @staticmethod
    def get_non_overlapping_interactions(interactions):
        remaining_interactions = interactions.copy()
        groups = []

        while len(remaining_interactions) > 0:
            group = []
            agent_in_group = {}

            i = 0

            while(i < len(remaining_interactions) and remaining_interactions[i].proactive_agent not in agent_in_group
                  and remaining_interactions[i].reactive_agent not in agent_in_group):
                group.append(remaining_interactions[i])
                agent_in_group[remaining_interactions[i].proactive_agent] = True
                agent_in_group[remaining_interactions[i].reactive_agent] = True
                i += 1

            remaining_interactions = remaining_interactions[len(group):]
            groups.append(group)

        return groups

    @staticmethod
    def service_display_interactions(gui, interactions, agents, imprisoned_agents):
        groups = ServiceGUI.get_non_overlapping_interactions(interactions)
        for g in groups:
            gui.display_interaction(interactions,agents,imprisoned_agents)
            gui.display(0.5,30)
            gui.clear_interactions()
            gui.refresh()

    @staticmethod
    def service_display_mining(gui, agent_earnings_list, prison_agents):
        gui.clear_prison_components()
        gui.display_mining(agent_earnings_list,prison_agents,30,2)

    @staticmethod
    def service_update_hierarchy_request(gui, agentToWealth):
        gui.set_social_hierarchy(agentToWealth)

    @staticmethod
    def service_update_personalities_request(gui, agents):
        #x = "Giving " + str(agents)
        #print(x)
        gui.set_agent_personality_list(agents)

    def find_unserviced_request(self):
        self.lock.acquire()
        if len(self.requestsList) == 0:
            self.lock.release()
            return None
        self.lock.release()
        return self.requestsList[0]

    @staticmethod
    def service_display_title(gui, title):
        gui.display_title(title)

    @staticmethod
    def service_display_round(gui, round):
        gui.display_round(round)

    @staticmethod
    def service_display_total(gui, total):
        gui.display_total(total)

    @staticmethod
    def process_request(gui, request):
        if request is not None:
            code = request['type']
            args = request['args']

            if code == 1:
                # interactions
                ServiceGUI.service_display_interactions(gui,args[0], args[1], args[2])
            elif code == 2:
                # mining
                ServiceGUI.service_display_mining(gui,args[0], args[1])
            elif code == 3:
                # wealth
                ServiceGUI.service_update_hierarchy_request(gui,args[0])
                # self.gui.display(0.1, 30)
                gui.refresh(30)
            elif code == 4:

                ServiceGUI.service_update_personalities_request(gui,args[0])
                # self.gui.display(0.1, 30)
                gui.refresh(30)
            elif code == 5:
                ServiceGUI.service_display_title(gui,args[0])
                gui.display(0.7,30)
            elif code == 6:
                ServiceGUI.service_display_round(gui,args[0])
                gui.refresh(30)
            elif code == 7:
                ServiceGUI.service_display_total(gui,args[0])
                gui.refresh(30)

    def run(self):
        while True:
            request = self.find_unserviced_request()
            self.process_request(self.gui,request)
            if request is not None:
                self.requestsList.remove(request)
