from sklearn.linear_model import LogisticRegression
import numpy as np

class wvRN():

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
                        sum_N += weight * neigh['class']
                    elif self.use_probabilities and 'probability' in neigh.attributes() and neigh['probability'] is not None:
                        sum_N += weight * neigh['probability']
                    elif not self.use_probabilities and 'pred' in neigh.attributes() and neigh['pred'] is not None:
                        if neigh['pred'] == '+':
                            le_class = 1
                        else:
                            le_class = 0
                        sum_N += weight * le_class

                    prob_pos = sum_N / float(Z)
                    prob_neg = 1 - prob_pos

        return (prob_pos, prob_neg)

class nBC():

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
        self.prior['+'] = pos / float(len(nodes))
        self.prior['-'] = neg / float(len(nodes))

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
        self.cpd = {0: {'+': (counts['+'][0] + 1) / float(counts['+'][0] + counts['+'][1] + 2),
                        '-': (counts['-'][0] + 1) / float(counts['-'][0] + counts['-'][1] + 2)},
                    1: {'+': (counts['+'][1] + 1) / float(counts['+'][0] + counts['+'][1] + 2),
                        '-': (counts['-'][1] + 1) / float(counts['-'][0] + counts['-'][1] + 2)}}

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

        if np.isfinite(prob_pos) and np.isfinite(prob_neg):
            # Normalize pseudo-probabilities to sum up to 1
            prob_total = float(prob_pos + prob_neg)
            prob_pos = prob_pos / prob_total
            prob_neg = prob_neg / prob_total

        return (prob_pos, prob_neg)

class lrRN():

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

            # if (self.use_proportions):
            #     pos_neighs = pos_neighs / float(len(neighbors))
            #     neg_neighs = neg_neighs / float(len(neighbors))
            #
            # x.append([pos_neighs, neg_neighs])

            pos_neighs_prop = pos_neighs / float(len(neighbors))
            neg_neighs_prop = neg_neighs / float(len(neighbors))

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

        #     if (self.use_proportions):
        #         pos_neighs = pos_neighs / float(len(neighbors))
        #         neg_neighs = neg_neighs / float(len(neighbors))


        pos_neighs_prop = pos_neighs / float(len(neighbors))
        neg_neighs_prop = neg_neighs / float(len(neighbors))

        x.append([pos_neighs, neg_neighs, pos_neighs_prop, neg_neighs_prop])

        probability = self.model.predict_proba(np.array(x))

        return (probability[0][1], probability[0][0])
