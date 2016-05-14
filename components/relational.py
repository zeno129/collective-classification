from sklearn.linear_model import LogisticRegression
import numpy as np

class wvRN():

    prior = None

    use_probabilities = True
    probability_threshold = 0.5

    use_previous_step = False

<<<<<<< HEAD
    use_weights = False
=======
    test = False
>>>>>>> 719f255ae497ca4e26a5671263645f869cdf8d06

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
            if node.index != neigh.index:
                edge = graph.es.select(_within=[node.index, neigh.index])

                if self.use_weights and 'weight' in edge.attributes():
                    # Get edge weights
                    weight = edge['weight']

                    if type(weight) is list:
                        if len(weight) >= 1:
                            weight = weight[0]
                else:
                    # Still need a default weight for the formula
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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class cdRN():
    def __init__(self):
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class nBC():

    cpd = None
    prior = None

    use_probabilities = True
    probability_threshold = 0.5

<<<<<<< HEAD
    use_weights = False
=======
    test = False
>>>>>>> 719f255ae497ca4e26a5671263645f869cdf8d06

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
                    if node.index != neigh.index:
                        edge = graph.es.select(_within=[node.index, neigh.index])

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

                if node.index in neighbors.indices:
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
                edge = graph.es.select(_within=[node.index, neigh.index])

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



        return (prob_pos, prob_neg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

class nLB():
    def __init__(self):
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

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

        # Get neighbors
        neighbors = graph.vs.select(graph.neighborhood(node))

        for neigh in neighbors:
<<<<<<< HEAD
            if node.index != neigh.index:
                # (re-) Initialize counts per neighbor
                pos_neighs = 0
                neg_neighs = 0

=======
            # (re-) Initialize counts per neighbor
            pos_neighs = 0
            neg_neighs = 0

            if self.test:
                # Use ground truth for test
                if neigh['class'] == '+':
                    pos_neighs += 1
                else:
                    neg_neighs += 1
            else:
>>>>>>> 719f255ae497ca4e26a5671263645f869cdf8d06
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
<<<<<<< HEAD

                if (self.use_proportions):
                    pos_neighs = pos_neighs / float(len(neighbors))
                    neg_neighs = neg_neighs / float(len(neighbors))
=======

        #     # TODO: this might have caused a bug!!! :(
        #     if (self.use_proportions):
        #         pos_neighs = pos_neighs / float(len(neighbors))
        #         neg_neighs = neg_neighs / float(len(neighbors))
        #
        # x.append([pos_neighs, neg_neighs])

        pos_neighs_prop = pos_neighs / float(len(neighbors))
        neg_neighs_prop = neg_neighs / float(len(neighbors))
>>>>>>> 719f255ae497ca4e26a5671263645f869cdf8d06

        x.append([pos_neighs, neg_neighs, pos_neighs_prop, neg_neighs_prop])

        probability = self.model.predict_proba(np.array(x))

        return (probability[0][1], probability[0][0])
