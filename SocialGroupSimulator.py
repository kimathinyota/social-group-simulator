from src.Agent import *
from src.InformalSocialGroup import *
import random

social_group = SocialGroup()

def random_agent():
    bigfive = []
    for x in range(1, 6):
        temp = []
        for y in range(1, 7):
            temp.append(random.randrange(1, 34))
        bigfive.append(temp)
    return Agent(bigfive[0], bigfive[1], bigfive[2], bigfive[3], bigfive[4])


def run_agent(lock):
    global social_group
    agent = random_agent()
    agent.run(social_group, lock)


def start_social_group():

    global social_group
    social_group


    lock = threading.Lock()
    t1 = threading.Thread(target=run_agent, args=(lock,))
    t2 = threading.Thread(target=run_agent, args=(lock,))
    t1.start()
    t2.start()

