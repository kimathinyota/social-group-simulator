from src.InformalSocialGroup import *
from src.Agent import *
from src.RLearningAgent import *
import threading


def run_agent(lock, social_group, agent):
    agent.run(social_group, lock)


def add_agent(agent, social_group, lock):
    t1 = threading.Thread(target=run_agent, args=(lock, social_group, agent))
    t1.start()


def start_social_group(gui_server, gui_server_lock):
    social_group = SocialGroup(gui_server, gui_server_lock)
    lock = threading.Lock()


    wanted_values = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'O5', 'C1', 'C4', 'C5']
    unwanted_values = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']

    #agent1 = Agent([10,10,10,10,10,10],[30,30,30,30,30,30],[30,30,30,30,30,30],[30,30,30,30,30,30],[30,30,30,30,30,30], "Best")

    #agent2 = Agent([7,12,10,10,10,10],[30,30,30,30,30,30],[30,30,30,30,30,30],[30,30,30,30,30,30],[30,30,30,30,30,30], "Best")

    agent1 = get_best_agent("Strong",wanted_values, unwanted_values)
    agent2 = get_best_agent("Weak", unwanted_values, wanted_values)
    agent3 = zero_r_agent("Smart")


    add_agent(agent1, social_group, lock)
    add_agent(agent3, social_group, lock)


def main():
    gui_server = ServiceGUI()
    lock = threading.Lock()
    t1 = threading.Thread(target=start_social_group, args=(gui_server, lock))
    t1.start()
    gui_server.run()


if __name__ == "__main__":
    main()


def random_agent(name):
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4], name)


