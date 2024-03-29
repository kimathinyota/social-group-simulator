from src.Display.SocialGroupGUI import *
import threading


class ServiceGUI:
    def __init__(self):
        self.requestsList = []
        self.gui = SocialGroupGUI(1300, 700)
        self.lock = threading.Lock()

    def service_request(self, request_code, args):
        request = {'type': request_code, 'args': args}
        self.lock.acquire()
        self.requestsList.append(request)
        self.lock.release()

    def service_draw_interaction_request(self, agents, interaction_name):
        self.gui.draw_interction(interaction_name, agents[0].name, agents[1].name)

    def service_end_interaction_request(self, interaction_name):
        self.gui.remove_interaction(interaction_name)

    def service_update_hierarchy_request(self, agentToStatus):
        self.gui.set_social_hierarchy(agentToStatus)

    def service_update_personalities_request(self, agents):
        self.gui.set_agent_personality_list(agents)

    def find_unserviced_request(self):
        self.lock.acquire()
        if len(self.requestsList) == 0:
            return None
        r = self.requestsList[0]
        self.lock.release()
        return r

    def run(self):
        while True:

            request = self.find_unserviced_request()

            if request is not None:
                code = request['type']
                args = request['args']
                if code == 1:
                    self.service_draw_interaction_request(args[0], args[1])
                    self.gui.refresh()
                    #self.gui.display(0.1, 30)
                elif code == 2:
                    self.service_end_interaction_request(args[0])
                    self.gui.refresh()
                    self.gui.display(0.1, 30)
                elif code == 3:
                    self.service_update_hierarchy_request(args[0])
                    self.gui.display(0.1, 30)
                elif code == 4:
                    self.service_update_personalities_request(args[0])
                    self.gui.display(0.1, 30)

                self.requestsList.remove(request)







