from src.Agent import *
from src.LearningAgent import *
from src.DisplayServer import *
from src.GoldMiningEnvironment import *
import threading
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import numpy as np
import scipy.stats
import json
import csv
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
        gui_requests, analysis, hierarchy = RunningSimulation.run_environment(environment,agents)
        if should_upload:
            RunningSimulation.upload_to_database(analysis)
        if should_display:
            RunningSimulation.display_gui(gui_requests)
        return gui_requests, hierarchy


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
    def unique_colours(n):
        colours = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "000000",
        "#800000", "#008000", "#000080", "#808000", "#800080", "#008080", "#808080",
        "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0", "#C0C0C0",
        "#400000", "#004000", "#000040", "#404000", "#400040", "#004040", "#404040",
        "#200000", "#002000", "#000020", "#202000", "#200020", "#002020", "#202020",
        "#600000", "#006000", "#000060", "#606000", "#600060", "#006060", "#606060",
        "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0", "#00A0A0", "#A0A0A0",
        "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0", "#E0E0E0"]
        return colours[0:n]

    @staticmethod
    def plot_3d(X, Y, Z, xlabel="X", ylabel="Y", zlabel="Z", is_x_discrete=True, should_include_lob=True, should_include_corr=True, show=True, should_draw=True):
        title = xlabel + " vs " + ylabel + " vs " + zlabel
        x_to_y_z = {}
        for i in range(len(X)):
            x = X[i]
            res = x_to_y_z[x] if x in x_to_y_z else [[], []]
            res[0].append(Y[i])
            res[1].append(Z[i])
            x_to_y_z[x] = res

        unique = list(x_to_y_z.keys())
        colours = Experiment.unique_colours(len(unique))
        x_to_colour = {unique[i]: colours[i] for i in range(len(unique))}

        if is_x_discrete:
            x_to_position = {unique[i - 1]: i for i in range(1, len(unique) + 1)}
        else:
            x_to_position = {x:x for x in unique}

        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            ax.set_title(title)

        x_to_correlation = {}

        for x in x_to_y_z:
            Y, Z, = x_to_y_z[x][0], x_to_y_z[x][1]
            p, color = x_to_position[x], x_to_colour[x]
            X = [p for i in range(len(Y))]

            should_include_lob = len(Y) >= 2 and len(Z) >= 2 and should_include_lob

            if should_draw:
                if should_include_lob:
                    ax.scatter(X, Y, Z,color=color)
                else:
                    ax.scatter(X, Y, Z, color=color, label=x)

            if should_include_lob:
                X, Y, Z = tuple(np.array(i) for i in [X, Y, Z])
                Xn,Yn, Zn = X.copy(), Y.copy(), Z.copy()
                if should_draw:
                    b, m = polyfit(Y, Z, 1)
                    Z = b + m * Y
                label = str(x)
                if should_include_corr:
                    r, rs, ps, pr = Experiment.get_correlation_data(Yn, Zn)
                    label += " " + str((rs,r,ps,pr))
                    x_to_correlation[x] = r, rs, ps, pr
                if should_draw:
                    ax.plot(X, Y, Z, color=color, label=label)

        if should_draw and ((not is_x_discrete) or should_include_corr):
            plt.legend(loc='upper left', title=xlabel + " ($r_s$, r, $p_s$, $p_r$)", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        if should_draw and show:
            plt.show()
        return x_to_correlation


    @staticmethod
    def get_correlation_data(X, Y):
        r, pr = scipy.stats.pearsonr(X, Y)
        rs, ps = scipy.stats.spearmanr(X, Y)
        rnd = 4
        rs, r = round(rs, rnd), round(r, rnd)
        ps, pr = round(ps, rnd), round(pr, rnd)
        return r, rs, ps, pr


    @staticmethod
    def plot_2d(X, Y, xlabel="X", ylabel="Y", show=True, should_draw=True):
        title = xlabel + " vs " + ylabel
        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        X, Y = tuple(np.array(i) for i in [X, Y])
        Xn, Yn = X.copy(), Y.copy()
        r, rs, ps, pr = Experiment.get_correlation_data(Xn, Yn)
        if should_draw:
            ax.scatter(X, Y)
            b, m = polyfit(X,Y,1)
            Y = b + m * X
            ax.plot(X, Y)
            txt = "$r_s$ = " + str(rs) + " (p=" + str(ps) + ")" + ", r = " + str(r) + " (p=" + str(pr) + ")"
            fig.text(0.01, 0.99, txt, ha='left', va='top',fontsize="large")
            if show:
                plt.show()
        return r, rs, ps, pr

    @staticmethod
    def plot_bar_chart(X, Y, xlabel="X", ylabel="Y",show=True, should_draw=True):
        title = xlabel + " vs " + ylabel
        if should_draw:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.bar(X, Y)
        if should_draw and show:
            plt.show()
        return fig

    @staticmethod
    def create_directory(dirName):
        try:
            # Create target Directory
            os.mkdir(dirName)
        except FileExistsError:
            pass

    @staticmethod
    def run_x_vs_earned_experiment(agent_var,total_number_of_agents, directory, starting=1, ending=9, number_of_repeats=3, number_of_rounds=40, starting_wealth=10000,
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
        total_runs = (ending - starting + 1) * number_of_repeats
        print("Running experiment for " + xl + " Total simulation runs: " + str(total_runs))
        count = 0

        for p in range(starting, ending+1):
            p /= 10
            experimental_agent = Experiment.change_agent_value(agent.copy(), agent_var, p)
            for i in range(number_of_repeats):
                experimental_agents = [agent.copy() for agent in ([experimental_agent] + other_agents)]
                RunningSimulation.simulate(experimental_agents, False, number_of_rounds,
                                           starting_wealth=starting_wealth, minimum_mining_amount=minimum_mining_amount)
                count += 1
                print("Finished " + xl + ": " + str(count) + "/" + str(total_runs) )
        # Data analysis
        # Get table: X | Round | Average-Wealth-Earnings

        title1 = xl + "vsGold"
        data1 = Experiment.construct_x_vs_gold_earned_json(agent_var,n,analysis)
        title2 = xl + "vsComp"
        data2 = Experiment.construct_x_vs_comp_earned_json(agent_var,n,analysis)
        title3 = xl + "vsInteractionChoices"
        data3 = Experiment.construct_x_vs_interaction_choice(agent_var,n,analysis)
        data = [(title1,data1), (title2, data2), (title3, data3)]

        variable_directory = directory + "/" + xl
        Experiment.create_directory(variable_directory)

        for d in data:
            t, d = d
            with open(variable_directory + "/" + t + '.json', 'w') as fp:
                json.dump(d, fp)

        print("Finished experiment for " + xl)


    @staticmethod
    def run_interaction_vs_earn(total_number_of_agents, directory, starting=1, ending=9, number_of_repeats=3, number_of_rounds=40, starting_wealth=10000,
                                   minimum_mining_amount=10):
        analysis = Analysis()
        analysis.remove_all_data()

        n = "testAgent"
        count = 0

        total_runs = (ending - starting + 1) * number_of_repeats
        print("Running Experiment for Interaction vs Earn: Total Simulations: " + str(total_runs))

        for p in range(starting, ending+1):
            other_agents = []
            for i in range(total_number_of_agents):
                name = "AG" + str(i)
                other_agents.append(Agent.random(name))

            for i in range(number_of_repeats):
                experimental_agents = [agent.copy() for agent in other_agents]
                RunningSimulation.simulate(experimental_agents, False, number_of_rounds,
                                           starting_wealth=starting_wealth, minimum_mining_amount=minimum_mining_amount)
                count += 1
                print(str(count) + "/" + str(total_runs))

        # Data analysis
        title1 = "InteractionVsGold"
        data1 = Experiment.construct_interaction_vs_gold_earned_json(analysis)

        with open(directory + "/" + title1 + '.json', 'w') as fp:
            json.dump(data1, fp)

        print("Finished Experiment for Interaction vs Earn")

    @staticmethod
    def construct_x_vs_gold_earned_json(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        x, y, z = analysis.get_x_vs_round_to_earn(agent_var, agent_name)
        yl, zl = "Round", "Gold Earned"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_earn(agent_var, agent_name)
        yl = "Total Gold Earned"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': [d1],
            '2D': [d2]
        }
        return data

    @staticmethod
    def construct_x_vs_interaction_choice(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        f, mp, mr, tp, tr, hp, hr = analysis.get_x_vs_interaction_choices(agent_var, agent_name)

        # x --> {f, mp, mr, tp, tr, hp, hr}
        var_to_values = {}

        X, Y, Z = [], [], []

        x, y = f
        for i in range(len(x)):
            X.append("Friendship")
            Y.append(x[i])
            Z.append(y[i])

        d1 = {"data": (x, y), "labels": (xl, "Friendship Total")}
        x, y = mp
        for i in range(len(x)):
            X.append("Mentorship\P")
            Y.append(x[i])
            Z.append(y[i])
        d2 = {"data": (x, y), "labels": (xl, "# Mentorship Proactive")}
        x, y = mr
        for i in range(len(x)):
            X.append("Mentorship\R")
            Y.append(x[i])
            Z.append(y[i])
        d3 = {"data": (x, y), "labels": (xl, "# Mentorship Reactive")}
        x, y = tp
        for i in range(len(x)):
            X.append("Theft\P")
            Y.append(x[i])
            Z.append(y[i])
        d4 = {"data": (x, y), "labels": (xl, "# Theft Proactive")}
        x, y = tr
        for i in range(len(x)):
            X.append("Theft\R")
            Y.append(x[i])
            Z.append(y[i])
        d5 = {"data": (x, y), "labels": (xl, "# Theft Reactive")}
        x, y = hp
        for i in range(len(x)):
            X.append("Help\P")
            Y.append(x[i])
            Z.append(y[i])
        d6 = {"data": (x, y), "labels": (xl, "# Help Proactive")}
        x, y = hr
        for i in range(len(x)):
            X.append("Help\R")
            Y.append(x[i])
            Z.append(y[i])
        d7 = {"data": (x, y), "labels": (xl, "# Help Reactive")}

        lst = [d1, d2, d3, d4, d5, d6, d7]

        data = {
            '3DB': [{"data": (X,Y,Z), "labels":("Interaction Type",Experiment.variable_to_name[agent_var], "Number of interactions")}]
        }

        return data

    @staticmethod
    def construct_x_vs_comp_earned_json(agent_var, agent_name, analysis=Analysis()):
        xl = Experiment.variable_to_name[agent_var]
        x, y, z = analysis.get_x_vs_round_to_comp(agent_var, agent_name)
        yl, zl = "Round", "Competency Improved"
        d1 = {"data": (x, y, z), "labels": (xl, yl, zl)}
        x, y = analysis.get_x_vs_total_comp(agent_var, agent_name)
        yl = "Total Competency Increase"
        d2 = {"data": (x, y), "labels": (xl, yl)}
        data = {
            '3D': [d1],
            '2D': [d2]
        }
        return data

    @staticmethod
    def construct_interaction_vs_gold_earned_json(analysis=Analysis()):
        labels, x, y = analysis.get_interaction_vs_round_to_earn()
        d1 = {"data":(labels,x,y), "labels":("Interaction Type","Round","Gold Earned")}

        x, y = analysis.get_interaction_vs_total_earn()
        d2 = {"data": (x, y), "labels": ("Interaction Type", "Total Gold Earned")}

        x, y = analysis.get_interaction_to_total()
        d3 = {"data": (x, y), "labels": ("Interaction Type", "Total Number of Rounds")}

        data = {
            '3DB': [d1],
            '2DB': [d2, d3]
        }
        return data



    @staticmethod
    def estimated_time_seconds(number_of_rounds):
        s = 32
        return s * number_of_rounds/40


    @staticmethod
    def run_all_experiments(total_number_of_agents, agent_variable_experiment_directory,
                            interaction_experiment_directory, start=1, end=10, repeats=3,
                            agent_variables=["CM", "PC", "PX", "CA", "PE", "PO", "PH", "PA"], number_of_rounds=40,
                            starting_wealth=10000, minimum_mining_amount=10):

        experimental_variables = agent_variables

        secs_per_sim = Experiment.estimated_time_seconds(number_of_rounds)

        number_sims = (end - start + 1) * repeats

        total_sims = (len(experimental_variables) + 1) * number_sims

        total_time = round((total_sims * secs_per_sim) / 60, 2)

        print("Running all experiments ... Estimated time: " + str(total_time) + " mins")
        print("...")

        for variable in experimental_variables:
            Experiment.run_x_vs_earned_experiment(variable,total_number_of_agents, agent_variable_experiment_directory,
                                                  start,end,repeats, number_of_rounds,starting_wealth,
                                                  minimum_mining_amount)

        Experiment.run_interaction_vs_earn(total_number_of_agents, interaction_experiment_directory, start, end, repeats)



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
            '3D': [d1],
            '2D': [d2]
        }
        with open(directory + "/" + xl + '.json', 'w') as fp:
            json.dump(data, fp)

    @staticmethod
    def load_json(data,show=True, csv_string="", should_draw=True):

        d1 = data['3D'] if '3D' in data else []
        d2 = data['2D'] if '2D' in data else []
        d3 = data['2DB'] if '2DB' in data else []
        d4 = data['3DB'] if '3DB' in data else []

        if not isinstance(d1,list):
            d1 = [d1]
        if not isinstance(d2,list):
            d2 = [d2]
        if not isinstance(d3,list):
            d3 = [d3]
        if not isinstance(d4,list):
            d4 = [d4]

        for d in d1:
            x, y, z = d['data']
            xl, yl, zl = d['labels']
            Experiment.plot_3d(x, y, z, xl, yl, zl,False, True, True, False,should_draw)

        for d in d2:
            x, y = d['data']
            xl, yl = d['labels']
            r, rs, ps, pr = Experiment.plot_2d(x, y, xl, yl,False,should_draw)
            line = Experiment.construct_line(xl,yl, r, pr, rs, ps)
            csv_string += line

        for d in d3:
            x, y = d['data']
            xl, yl = d['labels']
            Experiment.plot_bar_chart(x,y, xl, yl,False, should_draw)

        for d in d4:
            x, y, z = d['data']
            xl, yl, zl = d['labels']
            x_to_correlation = Experiment.plot_3d(x, y, z, xl, yl, zl,True, True, True, False,should_draw)
            for x in x_to_correlation:
                r, rs, ps, pr = x_to_correlation[x]
                line = Experiment.construct_line(yl, x, r, pr, rs, ps)
                csv_string += line

        if show:
            plt.show()
        return csv_string



    @staticmethod
    def construct_line(variable1, variable2, r, pr, rs, ps):
        line = variable1 + "," + variable2 + "," + str(rs) + "," + str(ps) + "," + str(r) + "," + str(pr)
        line += "\n"
        return line


    @staticmethod
    def load_graph(path_to_json, show=True, csv_string="", should_draw=True):
        data = Experiment.get_json(path_to_json)
        if data is not None:
            csv_string += Experiment.load_json(data, False, csv_string, should_draw)
        if show:
            plt.show()
        return csv_string

    @staticmethod
    def get_json(path_to_json):
        if ".json" in path_to_json:
            with open(path_to_json, 'r') as fp:
                data = json.load(fp)
                return data
        return None

    @staticmethod
    def load_graphs(directory, show=True, csv_string="", should_draw=True):
        entries = os.listdir(directory+"/")
        for entry in entries:
            e = directory+"/" + str(entry)
            csv_string += Experiment.load_graph(e, False, should_draw=should_draw)
        if show:
            plt.show()
        return csv_string

    @staticmethod
    def update_file(file_path, information):
        if not os.path.exists(file_path):
            with open(file_path, 'w'): pass
        f = open(file_path, "a")
        f.write(information)
        f.close()

    @staticmethod
    def display():
        plt.show()


    @staticmethod
    def show_experiment_results(agent_variable_directory, interaction_directory, agent_variables_to_display, should_display_interaction_graphs, file_path, show=True):
        csv = "Variable1,Variable2,rs,ps,r,pr\n"
        for var in agent_variables_to_display:
            direc = agent_variable_directory + "/" + Experiment.variable_to_name[var]
            print(direc)
            csv += Experiment.load_graphs(direc, False, should_draw=show)
        Experiment.update_file(file_path,csv)
        if should_display_interaction_graphs:
            Experiment.load_graphs(interaction_directory,False, should_draw=show)

        if show:
            plt.show()

    @staticmethod
    def get_expected_values(expected_csv_file_location):
        data = {}
        with open(expected_csv_file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    var1, var2, expected = row
                    data[(var1, var2)] = expected
                line_count += 1
            print(f'Processed {line_count} lines.')
        return data

    @staticmethod
    def get_actual_values(actual_csv_file_location):
        data = {}
        with open(actual_csv_file_location) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    var1, var2, rs, ps, r, p = row
                    data[(var1, var2)] = (rs, ps, r, p)
                line_count += 1
            print(f'Processed {line_count} lines.')
        return data

    @staticmethod
    def tofloat(x):
        f = float(x)
        f = 0 if math.isnan(f) else f
        return f


    @staticmethod
    def check_values(expected_csv_file_location, actual_csv_file_location, save_location, alpha=0.09, start_positive=0.2, start_negative=-0.2):
        expected_data = Experiment.get_expected_values(expected_csv_file_location)
        actual_data = Experiment.get_actual_values(actual_csv_file_location)

        Experiment.update_file(save_location, "Independent Variable, Dependent Variable, Spearman (rs), p, Pearson(r), "
                                              "p, Expected? \n")

        for variables in actual_data:
            var1, var2 = variables
            rsl, psl, rl, pl = actual_data[variables]
            expected = expected_data[variables] if variables in expected_data else '?'
            line = var1 + "," + var2 + "," + rsl + "," + psl + "," + rl + "," + pl + ","
            if expected != '?':
                rs, ps, r, p = Experiment.tofloat(rsl), Experiment.tofloat(psl), Experiment.tofloat(rl), Experiment.tofloat(pl)
                is_statistically_significant = ps <= alpha or p <= alpha
                extra = "SI" if not is_statistically_significant else ""

                matches = (expected == '1' and (rs >= start_positive or r >= start_positive)) \
                          or (expected == '-1' and (rs <= start_negative or r <= start_negative))

                result = "Y" if matches else "N"
                result += extra
                line += result
            line += "\n"
            Experiment.update_file(save_location, line)






































