from src.InformalSocialGroup import *
from src.Agent import *
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

    agent1 = get_best_agent("Best", wanted_values, unwanted_values)
    agent2 = get_best_agent("Random", [], [])

    add_agent(agent1, social_group, lock)
    add_agent(agent2, social_group, lock)


def main():
    gui_server = ServiceGUI()
    lock = threading.Lock()
    t1 = threading.Thread(target=start_social_group, args=(gui_server, lock))
    t1.start()
    gui_server.run()


if __name__ == "__main__":
    main()

