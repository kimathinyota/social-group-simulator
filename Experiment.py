from src.Agent import *
from src.DisplayServer import *
from src.GoldMiningEnvironment import *
import threading
import matplotlib.pyplot as plt
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


class RunningSimulation:

    @staticmethod
    def run_agent(agent, environment):
        agent.set_environment(environment)
        agent.run()

    @staticmethod
    def add_agent(agent, environment):
        t1 = threading.Thread(target=RunningSimulation.run_agent, args=(agent, environment))
        t1.start()

    @staticmethod
    def setup_environment(environment, agents):
        for agent in agents:
            RunningSimulation.add_agent(agent, environment)

    @staticmethod
    def run_environment(environment, agents):
        RunningSimulation.setup_environment(environment, agents)
        return environment.run()

    @staticmethod
    def display_gui(gui_requests):
        gui = SocialGroupGUI(1400, 800)
        for request in gui_requests:
            ServiceGUI.process_request(gui, request)

    @staticmethod
    def upload_to_database(analysis):
        analysis.query_all()

    @staticmethod
    def simulate(agents, should_display, number_of_rounds=40, should_upload=True, should_test=False,
                 starting_wealth=10000, minimum_mining_amount=10):
        environment = ResourceMiningEnvironment(starting_wealth,minimum_mining_amount,number_of_rounds,should_test)
        gui_requests, analysis = RunningSimulation.run_environment(environment,agents)
        if should_upload:
            RunningSimulation.upload_to_database(analysis)
        if should_display:
            RunningSimulation.display_gui(gui_requests)


class Experiment:

    # Variables:
    #     Agent
    #         Personality
    #             PH: Honesty-Humility
    #             PE: Emotionality
    #             PX: Extraversion
    #             PA: Agreeableness
    #             PC: Conscientiousness
    #             PO: Openeness
    #         Competence
    #             CM: Mining
    #             CA: Appraisal

    variable_to_name = {
        "PH": "Honesty-Humility",
        "PE": "Emotionality",
        "PX": "Extraversion",
        "PA": "Agreeableness",
        "PC": "Conscientiousness",
        "PO": "Openeness",
        "CM": "Mining",
        "CA": "Appraisal"
    }

    @staticmethod
    def is_variable(variable):
        return variable in list(Experiment.variable_to_name.keys())

    @staticmethod
    def is_agent_variable(variable):
        f = variable[0]
        return Experiment.is_variable(variable) and (f=='P' or f=='C')

    @staticmethod
    def change_agent_value(agent, agent_variable, percentage):
        if not Experiment.is_agent_variable(agent_variable):
            return agent

        f, s = agent_variable[0], agent_variable[1]
        if f == 'P':
            val = percentage * agent.personality_template.max_facet_score
            agent.personality_template.set_all_facets_in_dimension(agent.personality, s, val)
        elif agent_variable == "CM":
            agent.competency.update(percentage, agent.competency.appraisal_skill)
        elif agent_variable == "CA":
            agent.competency.update(agent.competency.mining_skill, percentage)

        return agent



    @staticmethod
    def plot_3d(X, Y, Z, xlabel="X", ylabel="Y", zlabel="Z"):
        title = xlabel + " vs " + ylabel + " vs " + zlabel
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = tuple(np.array([i]) for i in [X, Y, Z])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.plot_wireframe(X, Y, Z)
        return fig

    @staticmethod
    def plot_2d(X, Y, xlabel="X", ylabel="Y"):
        title = xlabel + " vs " + ylabel
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        X, Y = tuple(np.array(i) for i in [X, Y])
        ax.scatter(X, Y)
        return fig

    @staticmethod
    def run_x_vs_earned_experiment(agent_var,total_number_of_agents, directory, number_of_rounds=40, starting_wealth=10000,
                                   minimum_mining_amount=10):
        if not Experiment.is_agent_variable(agent_var):
            print("Experiment couldn't start - input variable isn't an agent variable")
            return None
        analysis = Analysis()
        analysis.remove_all_data()
        other_agents = []
        for i in range(total_number_of_agents - 1):
            name = "AG" + str(i)
            other_agents.append(Agent.random(name))
        n = "testAgent"
        agent = Agent.random(n)
        xl = Experiment.variable_to_name[agent_var]
        print("Simulating for " + xl)
        count = 0
        for p in range(1, 9):
            p /= 10
            experimental_agent = Experiment.change_agent_value(agent.copy(), agent_var, p)
            for i in range(3):
                experimental_agents = [agent.copy() for agent in ([experimental_agent] + other_agents)]
                RunningSimulation.simulate(experimental_agents, False, number_of_rounds,
                                           starting_wealth=starting_wealth, minimum_mining_amount=minimum_mining_amount)
                count += 1
                print(xl + ": " + str(count) + "/" + "27")
        # Data analysis
        # Get table: X | Round | Average-Wealth-Earnings
        x, y, z = analysis.get_x_vs_round_to_earn(agent_var, n)
        yl, zl = "Round", "Gold Earned"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_earn(agent_var, n)
        yl = "Total Gold Earned"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': d1,
            '2D': d2
        }
        with open(directory + "/" + xl + '.json', 'w') as fp:
            json.dump(data, fp)

    @staticmethod
    def run_all_earned_wealth_experiments(total_number_of_agents, directory, number_of_rounds=40, starting_wealth=10000,
                                          minimum_mining_amount=10):
        for variable in Experiment.variable_to_name:
            if Experiment.is_agent_variable(variable):
                Experiment.run_x_vs_earned_experiment(variable,total_number_of_agents,directory,number_of_rounds,starting_wealth,
                                                      minimum_mining_amount)




    @staticmethod
    def plot(agent_var, directory):
        n = "testAgent"
        analysis = Analysis()
        # Data analysis
        # Get table: X | Round | Average-Wealth-Earnings
        x, y, z = analysis.get_x_vs_round_to_earn(agent_var, n)
        xl, yl, zl = Experiment.variable_to_name[agent_var], "Round", "Gold Earned"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_earn(agent_var, n)
        yl = "Total Gold Earned"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': d1,
            '2D': d2
        }
        with open(directory + "/" + xl + '.json', 'w') as fp:
            json.dump(data, fp)

    @staticmethod
    def load_json(data):
        d1 = data['3D']
        d2 = data['2D']
        x, y, z = d1['data']
        xl, yl, zl = d1['labels']
        Experiment.plot_3d(x, y, z, xl, yl, zl)
        x, y = d2['data']
        xl, yl = d2['labels']
        Experiment.plot_2d(x, y, xl, yl)

    @staticmethod
    def load_graph(path_to_json, show=True):
        if ".json" in path_to_json:
            with open(path_to_json, 'r') as fp:
                data = json.load(fp)
                Experiment.load_json(data)
        if show:
            plt.show()

    @staticmethod
    def load_graphs(directory):
        entries = os.listdir(directory+"/")
        for entry in entries:
            e = directory+"/" + str(entry)
            Experiment.load_graph(e, False)
        plt.show()





















