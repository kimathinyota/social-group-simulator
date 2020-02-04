import operator
import random
from src.graphics import *


class SocialGroupGUI:

    def __init__(self, width, height):
        self.win = GraphWin('Social Group', width, height, autoflush=False)
        x = 0.8 * width
        y = 0.7 * height

        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.agent_canvas = {}
        self.hierarchy_canvas = {}
        self.personality_canvas = []

        agent_view = Rectangle(Point(0, 0), Point(x, y))
        agent_view.setFill(color_rgb(143, 170, 220))
        agent_view.setWidth(2)
        agent_view.setOutline(color_rgb(32, 56, 100))
        agent_view.draw(self.win)

        hierarchy_view = Rectangle(Point(x, 0), Point(width, y))
        hierarchy_view.setFill(color_rgb(0, 176, 240))
        hierarchy_view.setWidth(2)
        hierarchy_view.setOutline(color_rgb(32, 56, 100))
        hierarchy_view.draw(self.win)

        label = Text(Point(int((x + width) / 2), int(0.02 * height + 10)), "Social Hierarchy")
        label.setSize(23)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)
        self.hierarchy_starting_height = 50
        self.hierarchy_separation = 15
        self.hierarchy_box_height = 30

        personality_view = Rectangle(Point(0, y), Point(width, height))
        personality_view.setFill(color_rgb(189, 215, 238))
        personality_view.setWidth(2)
        personality_view.setOutline(color_rgb(32, 56, 100))
        personality_view.draw(self.win)

        labelWidth = 95
        labelHeight = 30
        label = Text(Point(int(0.02 * width + labelWidth / 2), int(0.025 * height + y)), "Personality")
        label.setSize(23)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)

        desc = 'Agent | (N1,N2,N3,N4,N5,N6) | (C1,C2,C3,C4,C5,C6) | (A1,A2,A3,A4,A5,A6) | (E1,E2,E3,E4,E5,E6) | (O1,O2,O3,O4,O5,O6)'
        labelWidth = 10 + self.estimate_length(desc)
        label = Text(Point(int(10 + labelWidth / 2), int(0.025 * height + y + labelHeight)), desc)
        label.setSize(17)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("normal")
        label.draw(self.win)

        self.personality_starting_height = int(0.025 * height + y + labelHeight + 25)
        self.personality_current_height = self.personality_starting_height

    def estimate_length(self, text):
        return len(text) * 8

    def draw_agent_personality(self, name, personality):
        order = ['N', 'C', 'A', 'E', 'O']
        text = name + ' | '
        for j in range(0, 5):
            c = order[j]
            dimension = '('
            for i in range(1, 7):
                p = personality[c + str(i)]
                t = '0' + str(p) if p < 10 else str(p)
                dimension += t
                if i < 6:
                    dimension += ","
            dimension += ')'
            text += dimension
            if j < 4:
                text += ' | '

        labelWidth = self.estimate_length(text)
        label = Text(Point(int(labelWidth / 2), self.personality_current_height), text)
        label.setSize(17)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("normal")
        label.draw(self.win)
        self.personality_canvas.append(label)
        self.personality_current_height += 25

    def close(self):
        self.win.close()

    def clear_agent_canvas(self):
        for key in self.agent_canvas:
            self.remove_components(self.agent_canvas[key])
        self.agent_canvas.clear()

    def clear_personality_canvas(self):
        for item in self.personality_canvas:
            item.undraw()
        self.personality_canvas.clear()
        self.personality_current_height = self.personality_starting_height

    def display(self, secs, rate):
        for i in range(1, int(secs * rate)):
            update(rate)


    def draw_interction(self, key, first, second):

        components = []

        w = 0.6 * self.x
        s = 0.2 * self.x
        x1 = random.randrange(s, int((s + w / 2)))

        h = 0.6 * self.y
        s2 = 0.2 * self.y
        y1 = random.randrange(s2, int(s2 + h / 2))

        w2 = 0.8 * self.x
        x2 = random.randrange(int(s + w / 2), w2)

        h2 = 0.8 * self.y
        y2 = random.randrange(int(s2 + h / 2), h2)

        line = Line(Point(x1, y1), Point(x2, y2))
        line.setFill(color_rgb(237, 125, 49))
        line.setWidth(5)

        line.draw(self.win)
        components.append(line)

        ag1 = self.draw_agent(first, x1, y1)
        ag2 = self.draw_agent(second, x2, y2)

        components = components + ag1 + ag2

        self.agent_canvas[key] = components







    def draw_agent(self, name, x, y):

        components = []

        circle = Circle(Point(x, y), 35)
        circle.setWidth(2)
        circle.setFill(color_rgb(255, 192, 0))
        circle.setOutline(color_rgb(155, 140, 0))
        circle.draw(self.win)
        components.append(circle)

        label = Text(Point(x, y), name)
        label.setSize(18)
        label.setTextColor("black")
        label.setFace("arial")
        label.setStyle("bold")
        label.draw(self.win)
        components.append(label)

        return components

    def draw_hierarchy_box(self, name, status, position):

        sx = 10 + self.x
        ex = self.width - 10

        sy = self.hierarchy_starting_height + position * (self.hierarchy_box_height + self.hierarchy_separation)
        ey = sy + self.hierarchy_box_height

        number_width = 30
        x1 = sx + number_width
        x2 = x1 + ((ex - sx) - number_width) / 2

        components = []
        if position in self.hierarchy_canvas:
            self.remove_components(self.hierarchy_canvas[position])
            self.hierarchy_canvas.pop(position, None)

        left = Rectangle(Point(sx, sy), Point(x1, ey))
        left.setFill('white')
        left.setWidth(2)
        left.setOutline('black')
        left.draw(self.win)
        components.append(left)

        rank = Text(Point(sx + int((x1 - sx) / 2), sy + int((ey - sy) / 2)), str(position + 1))
        rank.setSize(18)
        rank.setTextColor(color_rgb(237, 125, 49))
        rank.setFace("arial")
        rank.setStyle("bold")
        rank.draw(self.win)
        components.append(rank)

        middle = Rectangle(Point(x1, sy), Point(x2, ey))
        middle.setFill('white')
        middle.setWidth(2)
        middle.setOutline('black')
        middle.draw(self.win)
        components.append(middle)

        nameLabel = Text(Point(x1 + int((x2 - x1) / 2), sy + int((ey - sy) / 2)), name)
        nameLabel.setSize(18)
        nameLabel.setTextColor("black")
        nameLabel.setFace("arial")
        nameLabel.setStyle("bold")
        nameLabel.draw(self.win)
        components.append(nameLabel)

        right = Rectangle(Point(x2, sy), Point(ex, ey))
        right.setFill('white')
        right.setWidth(2)
        right.setOutline('black')
        right.draw(self.win)
        components.append(right)

        statusLabel = Text(Point(x2 + int((ex - x2) / 2), sy + int((ey - sy) / 2)), status)
        statusLabel.setSize(18)
        statusLabel.setTextColor("black")
        statusLabel.setFace("arial")
        statusLabel.setStyle("bold")
        statusLabel.draw(self.win)
        components.append(statusLabel)

        self.hierarchy_canvas[position] = components

    def clear_hierarchy_canvas(self):
        for p in self.hierarchy_canvas:
            self.remove_components(self.hierarchy_canvas[p])
            self.hierarchy_canvas[p].clear()

    def remove_interaction(self, key):
        if key not in self.agent_canvas:
            pass
        self.remove_components(self.agent_canvas[key])

    def remove_components(self, components):
        for x in components:            x.undraw()

    def set_social_hierarchy(self, agentToStatus):
        sorted_x = sorted(agentToStatus.items(), key=operator.itemgetter(1))
        position = 0
        self.clear_hierarchy_canvas()
        sorted_x.reverse()

        for x in sorted_x:
            agent = x[0]
            status = x[1]
            self.draw_hierarchy_box(agent.name, status, position)
            position += 1

    def set_agent_personality_list(self, agents):
        self.clear_personality_canvas()
        for a in agents:
            self.draw_agent_personality(a.name, a.personality)




