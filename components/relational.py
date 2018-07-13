from sklearn.linear_model import LogisticRegression
import numpy as np

class wvRN:

    name = 'wvrn'

    prior = None

    use_probabilities = True
    probability_threshold = 0.5

    use_previous_step = False

    test = False

    use_weights = False

    def __init__(self):
        pass

    def learn(self, graph, nodes):
        pass

    def predict(self, graph, node):
        Z = 0
        sum_N = 0

        # Init probabilities
        prob_pos = self.prior
        prob_neg = 1 - self.prior

        # Get neighbors
        neighbors = graph.vs.select(graph.neighborhood(node))

        for neigh in neighbors:
            # if node.index != neigh.index:
                # edge = graph.es.select(_within=[node.index, neigh.index])

            # Use assigned ids from indices -- subgraph selection could change indices!
            # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
            if node['id'] != neigh['id']:
                edge = graph.es.select(_within=[node['id'], neigh['id']])

                if self.use_weights and 'weight' in edge.attributes():
                    weight = edge['weight']

                    if type(weight) is list:
                        if len(weight) >= 1:
                            weight = weight[0]

                else:
                    weight = 1

                if weight != 0:
                    # Update normalizer
                    Z += weight

                    if neigh['use_label'] == True:
                        if neigh['class']:
                            sum_N += weight * neigh['class']
                    elif self.use_probabilities and 'probability' in neigh.attributes() and neigh['probability'] is not None:
                        sum_N += weight * neigh['probability']
                    elif not self.use_probabilities and 'pred' in neigh.attributes() and neigh['pred'] is not None:
                        if neigh['pred'] == '+':
                            le_class = 1
                        else:
                            le_class = 0
                        sum_N += weight * le_class

                    prob_pos = sum_N / np.longdouble(Z)
                    prob_neg = 1 - prob_pos

        return (prob_pos, prob_neg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
class nBC:

    name = 'nbc'

    cpd = None
    prior = None

    use_probabilities = True
    probability_threshold = 0.5

    test = False

    use_weights = False

    def __init__(self):
        pass

    def learn(self, graph, nodes):
        '''
        Learn pseudo-likelihood and priors for relational Naive Bayes.

        :param nodes: iGraph nodes
        :type nodes: node sequence / list (?)
        '''
        # Init prior & likelihood
        counts = {'+': {0: 0, 1: 0}, '-': {0: 0, 1: 0}}
        self.prior = {'+': 0, '-': 0}

        # Calc prior
        pos = nodes['class'].count(1)
        neg = nodes['class'].count(0)
        self.prior['+'] = pos / np.longdouble(len(nodes))
        self.prior['-'] = neg / np.longdouble(len(nodes))

        # Calc pseudo-likelihood
        for node in nodes:

            # Get neighbors
            neighbors = graph.vs.select(graph.neighborhood(node))

            if self.use_weights:
                for neigh in neighbors:
                    # Use assigned ids from indices -- subgraph selection could change indices!
                    # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
                    if node['id'] != neigh['id']:
                        edge = graph.es.select(_within=[node['id'], neigh['id']])

                        if self.use_weights and 'weight' in edge.attributes():
                            weight = edge['weight']

                            if type(weight) is list:
                                if len(weight) >= 1:
                                    weight = weight[0]
                        else:
                            weight = 1

                        if weight == 1:
                            if node['class'] == 1:
                                counts['+'][neigh['class']] += 1
                            elif node['class'] == 0:
                                counts['-'][neigh['class']] += 1

            else:
                if node['class'] == 1:
                    counts['+'][1] += neighbors['class'].count(1)
                    counts['+'][0] += neighbors['class'].count(0)
                elif node['class'] == 0:
                    counts['-'][1] += neighbors['class'].count(1)
                    counts['-'][0] += neighbors['class'].count(0)

                # Avoid any self-loops
                # Use assigned ids from indices -- subgraph selection could change indices!
                if node['id'] in neighbors['id']:
                    if node['class'] == 1:
                        counts['+'][1] -= 1
                    elif node['class'] == 0:
                        counts['-'][0] -= 1

        # Add Laplace correction
        self.cpd = {0: {'+': (counts['+'][0] + 1) / np.longdouble(counts['+'][0] + counts['+'][1] + 2),
                        '-': (counts['-'][0] + 1) / np.longdouble(counts['-'][0] + counts['-'][1] + 2)},
                    1: {'+': (counts['+'][1] + 1) / np.longdouble(counts['+'][0] + counts['+'][1] + 2),
                        '-': (counts['-'][1] + 1) / np.longdouble(counts['-'][0] + counts['-'][1] + 2)}}

        return self.prior


    def predict(self, graph, node):
        '''
        Predict class probabilities.

        :param node: node to classify
        :type node: iGraph Vertex
        '''

        # Init probabilities
        prob_pos = self.prior['+']
        prob_neg = self.prior['-']

        # Get neighbors
        neighbors = graph.vs.select(graph.neighborhood(node))

        if self.use_weights:
            for neigh in neighbors:
                # Use assigned ids from indices -- subgraph selection could change indices!
                # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
                if node['id'] != neigh['id']:
                    edge = graph.es.select(_within=[node['id'], neigh['id']])

                    if self.use_weights and 'weight' in edge.attributes():
                        weight = edge['weight']

                        if type(weight) is list:
                            if len(weight) >= 1:
                                weight = weight[0]
                    else:
                        weight = 1

                    if weight == 1:
                        if neigh['use_label'] == True:
                            prob_neg *= self.cpd[neigh['class']]['-']
                            prob_pos *= self.cpd[neigh['class']]['+']
                        elif 'pred' in neigh.attributes() and neigh['pred'] is not None:
                            if neigh['pred'] == '+':
                                le_class = 1
                            else:
                                le_class = 0
                            prob_neg *= self.cpd[le_class]['-']
                            prob_pos *= self.cpd[le_class]['+']
                        elif 'init' in neigh.attributes() and neigh['init'] is not None:
                            if neigh['init'] == '+':
                                le_class = 1
                            else:
                                le_class = 0
                            prob_neg *= self.cpd[le_class]['-']
                            prob_pos *= self.cpd[le_class]['+']
        else:
            for neigh in neighbors:
                # Use assigned ids from indices -- subgraph selection could change indices!
                if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
                    if neigh['use_label'] == True:
                        prob_neg *= self.cpd[neigh['class']]['-']
                        prob_pos *= self.cpd[neigh['class']]['+']
                    elif 'pred' in neigh.attributes() and neigh['pred'] is not None:
                        if neigh['pred'] == '+':
                            le_class = 1
                        else:
                            le_class = 0
                        prob_neg *= self.cpd[le_class]['-']
                        prob_pos *= self.cpd[le_class]['+']
                    elif 'init' in neigh.attributes() and neigh['init'] is not None:
                        if neigh['init'] == '+':
                            le_class = 1
                        else:
                            le_class = 0
                        prob_neg *= self.cpd[le_class]['-']
                        prob_pos *= self.cpd[le_class]['+']

        # Note: If no neighbors were labeled, it will use priors from labeled nodes
        if np.isfinite(prob_pos) and np.isfinite(prob_neg):
            # Normalize pseudo-probabilities to sum up to 1
            prob_total = np.longdouble(prob_pos + prob_neg)
            prob_pos = prob_pos / prob_total
            prob_neg = prob_neg / prob_total

        return (prob_pos, prob_neg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class stacked_nBC:

    name = 'nbc'

    cpd = None
    prior = None

    use_probabilities = True
    probability_threshold = 0.5

    test = False

    use_weights = False

    def __init__(self):
        pass

    def learn(self, graph, nodes):
        '''
        Learn pseudo-likelihood and priors for relational Naive Bayes.

        :param nodes: iGraph nodes
        :type nodes: node sequence / list (?)
        '''
        # Init prior & likelihood
        # TODO: Use WV probabilities (binned) as individual and relational features
        counts = {1: {'0.2': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.4': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.6': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.8': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '1.0': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}}},
                  0: {'0.2': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.4': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.6': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.8': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '1.0': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}}}}
        self.prior = {'+': 0, '-': 0}

        # Calc prior
        pos = nodes['class'].count(1)
        neg = nodes['class'].count(0)
        self.prior['+'] = pos / np.longdouble(len(nodes))
        self.prior['-'] = neg / np.longdouble(len(nodes))

        # Calc pseudo-likelihood
        for node in nodes:

            # Get neighbors
            neighbors = graph.vs.select(graph.neighborhood(node))

            # if self.use_weights:
            #     for neigh in neighbors:
            #         # Use assigned ids from indices -- subgraph selection could change indices!
            #         # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
            #         if node['id'] != neigh['id']:
            #             edge = graph.es.select(_within=[node['id'], neigh['id']])
            #
            #             if self.use_weights and 'weight' in edge.attributes():
            #                 weight = edge['weight']
            #
            #                 if type(weight) is list:
            #                     if len(weight) >= 1:
            #                         weight = weight[0]
            #             else:
            #                 weight = 1
            #
            #             if weight == 1:
            #                 if node['class'] == 1:
            #                     counts['+'][neigh['class']] += 1
            #                 elif node['class'] == 0:
            #                     counts['-'][neigh['class']] += 1
            #
            # else:

            # TODO: we're not using weights right now

            # get bin for node's WV+ICA feature
            bin_node = None
            if node['init'] <= 0.2:
                bin_node = '0.2'
            elif node['init'] <= 0.4:
                bin_node = '0.4'
            elif node['init'] <= 0.8:
                bin_node = '0.8'
            elif node['init'] <= 1.0:
                bin_node = '1.0'

            for neighbor in neighbors:
                # Avoid any self-loops
                # Use assigned ids from indices -- subgraph selection could change indices!
                if node['id'] != neighbor['id']:
                    # if neighbor['init'] != None:
                    # get bin for neighbor's WV+ICA feature
                    bin_neighbor = None
                    if neighbor['init'] <= 0.2:
                        bin_neighbor = '0.2'
                    elif neighbor['init'] <= 0.4:
                        bin_neighbor = '0.4'
                    elif neighbor['init'] <= 0.8:
                        bin_neighbor = '0.8'
                    elif neighbor['init'] <= 1.0:
                        bin_neighbor = '1.0'

                    # increase count according to neighbor bin and class
                    if neighbor['class']:
                        counts[node['class']][bin_node][bin_neighbor][neighbor['class']] += 1

        # Add Laplace correction
        # self.cpd = {0: {'+': (counts['+'][0] + 1) / np.longdouble(counts['+'][0] + counts['+'][1] + 2),
        #                 '-': (counts['-'][0] + 1) / np.longdouble(counts['-'][0] + counts['-'][1] + 2)},
        #             1: {'+': (counts['+'][1] + 1) / np.longdouble(counts['+'][0] + counts['+'][1] + 2),
        #                 '-': (counts['-'][1] + 1) / np.longdouble(counts['-'][0] + counts['-'][1] + 2)}}

        bins = ['0.2', '0.4', '0.6', '0.8', '1.0']

        self.cpd = {1: {'0.2': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.4': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.6': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.8': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '1.0': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}}},
                  0: {'0.2': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.4': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.6': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '0.8': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}},
                      '1.0': {'0.2': {0: 0, 1: 0}, '0.4': {0: 0, 1: 0}, '0.6': {0: 0, 1: 0}, '0.8': {0: 0, 1: 0}, '1.0': {0: 0, 1: 0}}}}

        for bin_node in bins:
            for bin_neigh in bins:
                # given neighbor class is 0, probability node is class 0
                self.cpd[0][bin_node][bin_neigh][0] = (counts[0][bin_node][bin_neigh][0] + 1) / \
                                                      np.longdouble(counts[0][bin_node][bin_neigh][0] +
                                                                    counts[1][bin_node][bin_neigh][0] + 2)
                # given neighbor class is 0, probability node is class 1
                self.cpd[1][bin_node][bin_neigh][0] = (counts[1][bin_node][bin_neigh][0] + 1) / \
                                                      np.longdouble(counts[0][bin_node][bin_neigh][0] +
                                                                    counts[1][bin_node][bin_neigh][0] + 2)

                # given neighbor class is 1, probability node is class 0
                self.cpd[0][bin_node][bin_neigh][1] = (counts[0][bin_node][bin_neigh][1] + 1) / \
                                                      np.longdouble(counts[0][bin_node][bin_neigh][1] +
                                                                    counts[1][bin_node][bin_neigh][1] + 2)
                # given neighbor class is 1, probability node is class 1
                self.cpd[1][bin_node][bin_neigh][1] = (counts[1][bin_node][bin_neigh][1] + 1) / \
                                                      np.longdouble(counts[0][bin_node][bin_neigh][1] +
                                                                    counts[1][bin_node][bin_neigh][1] + 2)


        return self.prior


    def predict(self, graph, node):
        '''
        Predict class probabilities.

        :param node: node to classify
        :type node: iGraph Vertex
        '''

        # Init probabilities
        prob_pos = self.prior['+']
        prob_neg = self.prior['-']

        # Get neighbors
        neighbors = graph.vs.select(graph.neighborhood(node))

        # if self.use_weights:
        #     for neigh in neighbors:
        #         # Use assigned ids from indices -- subgraph selection could change indices!
        #         # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
        #         if node['id'] != neigh['id']:
        #             edge = graph.es.select(_within=[node['id'], neigh['id']])
        #
        #             if self.use_weights and 'weight' in edge.attributes():
        #                 weight = edge['weight']
        #
        #                 if type(weight) is list:
        #                     if len(weight) >= 1:
        #                         weight = weight[0]
        #             else:
        #                 weight = 1
        #
        #             if weight == 1:
        #                 if neigh['use_label'] == True:
        #                     prob_neg *= self.cpd[neigh['class']]['-']
        #                     prob_pos *= self.cpd[neigh['class']]['+']
        #                 elif 'pred' in neigh.attributes() and neigh['pred'] is not None:
        #                     if neigh['pred'] == '+':
        #                         le_class = 1
        #                     else:
        #                         le_class = 0
        #                     prob_neg *= self.cpd[le_class]['-']
        #                     prob_pos *= self.cpd[le_class]['+']
        #                 elif 'init' in neigh.attributes() and neigh['init'] is not None:
        #                     if neigh['init'] == '+':
        #                         le_class = 1
        #                     else:
        #                         le_class = 0
        #                     prob_neg *= self.cpd[le_class]['-']
        #                     prob_pos *= self.cpd[le_class]['+']
        # else:

        # TODO: we're not using weights right now
        # get bin for node's WV+ICA feature
        bin_node = None
        if node['init'] <= 0.2:
            bin_node = '0.2'
        elif node['init'] <= 0.4:
            bin_node = '0.4'
        elif node['init'] <= 0.8:
            bin_node = '0.8'
        elif node['init'] <= 1.0:
            bin_node = '1.0'

        for neigh in neighbors:
            # get bin for neighbor's WV+ICA feature
            bin_neighbor = None
            if neigh['init'] <= 0.2:
                bin_neighbor = '0.2'
            elif neigh['init'] <= 0.4:
                bin_neighbor = '0.4'
            elif neigh['init'] <= 0.8:
                bin_neighbor = '0.8'
            elif neigh['init'] <= 1.0:
                bin_neighbor = '1.0'

            # Use assigned ids from indices -- subgraph selection could change indices!
            if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
                if neigh['use_label'] == True:
                    prob_neg *= self.cpd[0][bin_node][bin_neighbor][neigh['class']]
                    prob_pos *= self.cpd[1][bin_node][bin_neighbor][neigh['class']]
                elif 'pred' in neigh.attributes() and neigh['pred'] is not None:
                    if neigh['pred'] == '+':
                        le_class = 1
                    else:
                        le_class = 0
                    prob_neg *= self.cpd[0][bin_node][bin_neighbor][le_class]
                    prob_pos *= self.cpd[1][bin_node][bin_neighbor][le_class]
                # TODO: Use WV probabilities (binned) as individual and relational features
                # elif 'init' in neigh.attributes() and neigh['init'] is not None:
                #     if neigh['init'] == '+':
                #         le_class = 1
                #     else:
                #         le_class = 0
                #     prob_neg *= self.cpd[le_class]['-']
                #     prob_pos *= self.cpd[le_class]['+']

        # Note: If no neighbors were labeled, it will use priors from labeled nodes
        if np.isfinite(prob_pos) and np.isfinite(prob_neg):
            # Normalize pseudo-probabilities to sum up to 1
            prob_total = np.longdouble(prob_pos + prob_neg)
            prob_pos = prob_pos / prob_total
            prob_neg = prob_neg / prob_total

        return (prob_pos, prob_neg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class nLB:
    def __init__(self):
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class lrRN:

    name = 'log'

    test = False

    def __init__(self, use_proportions=False):
        self.model = LogisticRegression()
        self.use_proportions = use_proportions

    def learn(self, graph, nodes):
        '''
        Fit the model according to the given training data.

        :param nodes: iGraph nodes
        :type nodes: node sequence / list (?)
        '''

        x = []
        y = []

        for node in nodes:

            # Get neighbors
            neighbors = graph.vs.select(graph.neighborhood(node))

            # Get the counts
            pos_neighs = neighbors['class'].count(1)
            neg_neighs = neighbors['class'].count(0)

            # Avoid any self-loops
            # Use assigned ids from indices -- subgraph selection could change indices!
            if node['id'] in neighbors['id']:
                if node['class'] == 1:
                    pos_neighs -= 1
                elif node['class'] == 0:
                    neg_neighs -= 1

            pos_neighs_prop = pos_neighs / np.longdouble(len(neighbors))
            neg_neighs_prop = neg_neighs / np.longdouble(len(neighbors))

            x.append([pos_neighs, neg_neighs, pos_neighs_prop, neg_neighs_prop])

            y.append(node['class'])

        self.model.fit(np.array(x), np.array(y))

    def predict(self, graph, node):
        '''
        Predict class probabilities.

        :param node: node to classify
        :type node: iGraph Vertex
        '''

        x = []

        # Initialize counts
        pos_neighs = 0
        neg_neighs = 0

        # Get neighbors
        neighbors = graph.vs.select(graph.neighborhood(node))

        for neigh in neighbors:
            # Use assigned ids from indices -- subgraph selection could change indices!
            # if 'id' in node.attributes() and 'id' in neigh.attributes() and node['id'] != neigh['id']:
            if node['id'] != neigh['id']:
                # (re-) Initialize counts per neighbor
                # pos_neighs = 0
                # neg_neighs = 0

                if self.test:
                    # Use ground truth for test
                    if neigh['class'] == '+':
                        pos_neighs += 1
                    else:
                        neg_neighs += 1
                else:
                    if neigh['use_label'] == True:
                        if neigh['class'] == '+':
                            pos_neighs += 1
                        else:
                            neg_neighs += 1
                    elif 'pred' in neigh.attributes() and neigh['pred'] is not None:
                        if neigh['pred'] == '+':
                            pos_neighs += 1
                        else:
                            neg_neighs += 1
                    elif 'init' in neigh.attributes() and neigh['init'] is not None:
                        if neigh['init'] == '+':
                            le_class = 1
                        else:
                            le_class = 0

        pos_neighs_prop = pos_neighs / np.longdouble(len(neighbors))
        neg_neighs_prop = neg_neighs / np.longdouble(len(neighbors))

        x.append([pos_neighs, neg_neighs, pos_neighs_prop, neg_neighs_prop])

        probability = self.model.predict_proba(np.array(x))

        return (probability[0][1], probability[0][0])
