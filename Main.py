
from src.Agent import *
from src.DisplayServer import *
from src.GoldMiningEnvironment import *

import threading


def run_agent(agent, environment):
    agent.set_environment(environment)
    agent.run()


def add_agent(agent, environment):
    t1 = threading.Thread(target=run_agent, args=(agent, environment))
    t1.start()


def start_environment(gui_server):
    environment = ResourceMiningEnvironment(3000, 10, gui_server)

    agent1 = Agent.random("Kim")
    agent2 = Agent.random("Bob")
    agent3 = Agent.random("Tom")
    agent4 = Agent.random("Don")
    agent5 = Agent.random("Log")
    agent6 = Agent.random("Bro")

    add_agent(agent1, environment)
    add_agent(agent2, environment)
    add_agent(agent3, environment)
    add_agent(agent4, environment)
    add_agent(agent5, environment)
    add_agent(agent6, environment)

    environment.run()

def start_environment2(environment):
    agent1 = Agent.random("Kim")
    agent2 = Agent.random("Bob")
    agent3 = Agent.random("Tom")
    agent4 = Agent.random("Don")
    agent5 = Agent.random("Log")
    agent6 = Agent.random("Bro")
    agent7 = Agent.random("Gon")
    agent8 = Agent.random("Ken")
    agent9 = Agent.random("Ceb")
    agent10 = Agent.random("Dan")
    agent11 = Agent.random("Joe")
    agent12 = Agent.random("Yoe")

    add_agent(agent1, environment)
    add_agent(agent2, environment)
    add_agent(agent3, environment)
    add_agent(agent4, environment)
    add_agent(agent5, environment)
    add_agent(agent6, environment)
    add_agent(agent7, environment)
    add_agent(agent8, environment)
    add_agent(agent9, environment)
    add_agent(agent10, environment)
    add_agent(agent11, environment)
    add_agent(agent12, environment)


def main():
    gui_server = ServiceGUI()
    t1 = threading.Thread(target=start_environment, args=(gui_server,) )
    t1.start()
    gui_server.run()

def main2():
    environment = ResourceMiningEnvironment(10000, 10,40)
    start_environment2(environment)
    requests = environment.run()
    gui = SocialGroupGUI(1400, 800)

    for request in requests:
        ServiceGUI.process_request(gui, request)


def main3():
    environment = ResourceMiningEnvironment(10000, 10,5)
    start_environment2(environment)
    requests = environment.run()
    gui = SocialGroupGUI(1400, 800)



    agents = []
    prison = []
    interactions = []

    for request in requests:
        code = request['type']
        args = request['args']
        print("Request",request)
        if code == 1:
            interactions += args[0]
            agents = args[1]
            prison = args[2]
        else:
            # Display all interactions
            if len(interactions) > 0:
                new_request = ServiceGUI.construct_request(1,[interactions,agents,prison])
                print("Interaction request", new_request)
                ServiceGUI.process_request(gui, new_request)
                interactions = []
            ServiceGUI.process_request(gui,request)

    gui.display(100,30)



if __name__ == "__main__":
    main2()


def random_agent(name):
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(name,Competency.random(),HexacoPersonality().random_personality())


