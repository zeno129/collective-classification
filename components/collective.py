import random
import numpy as np


class IterativeClassification():
    '''Collective Inference Method: Iterative Classification'''

    def __init__(self):
        pass


class RelaxationLabeling():
    '''Collective Inference Method: Relaxation Labeling'''

    # Default values
    alpha = 0.99
    k = 1
    max_iterations = 1000

    parallel = False

    prior = None
    percent_labeled = 30

    def __init__(self):
        pass

    def learn(self, graph, nodes, relational):
        prior = relational.learn(graph, nodes)

        if prior:
            self.prior = prior
        else:
            # TODO: Take optional prior as an argument
            # Need this for wvRN (prior is not learned)
            self.prior = {'+': 0.5, '-': 0.5}
            relational.prior = self.prior['+']

    def predict(self, graph, nodes, relational):
        iterations = 0
        classes_changed = True

        # Step 1: Initialization  - - - - - - *
        # Init V^U with M_L -- don't have an M_L right now; use priors
        # tosses = [random.uniform(0, 1) for o in range(len(nodes))]
        # assignments = [['+'] if t <= int(self.prior['+']) else ['-']
        #                for t in tosses]
        # nodes['assignments'] = assignments

        if not self.parallel:
            # (4) Repeat for T = 100 iterations, or until no entities receive a new class label
            while (iterations < self.max_iterations and classes_changed):
                # Update num iterations
                iterations += 1

                # Save temporarily the previous labels for comparison
                if 'pred' in nodes.attributes():
                    prev_labels = nodes['pred']
                else:
                    prev_labels = []

                # Step 2: Random Ordering - - - - - - *
                ordering = range(len(nodes)) # will be less than nodes in entire graph
                random.shuffle(ordering)

                # Step 2.5: % labels bootstrap  - - - *
                if self.percent_labeled > 0:
                    # We will bootstrap from the nodes in the training set
                    if 'id' not in nodes.attributes():
                        graph.vs['id'] = range(len(graph.vs))
                    labeling = graph.vs.select(partition_eq=0)['id']
                    random.shuffle(labeling)
                    upto = len(nodes) * (self.percent_labeled / 100)
                    to_label = labeling[:upto + 1]

                    graph.vs.select(to_label)['use_label'] = True

                # Step 3: Iterate - - - - - - - - - - *
                for o in ordering:
                    prob_pos, prob_neg = relational.predict(graph, nodes[o])

                    if np.isfinite(prob_pos) and np.isfinite(prob_neg):
                        if (prob_pos + prob_neg) < 0.9:
                            # Scale (pseudo?) probabilities for coin flip
                            prob_total = float(prob_pos + prob_neg)
                            prob_pos = prob_pos / prob_total
                            prob_neg = prob_neg / prob_total

                    # Do new prediction
                    if prob_pos >= prob_neg:
                        label = '+'
                    else:
                        label = '-'

                    nodes[o]['probability'] = prob_pos
                    nodes[o]['pred'] = label

                if 'pred' in nodes.attributes() and nodes['pred'] == prev_labels:
                    classes_changed = False

            # for n in nodes:
            #     n['probability'] = n['prob']

            # Note: Probabilities are stored in graph


class GibbsSampling():
    '''Collective Inference Method: Gibbs Sampling'''

    # Default values
    burn_in_iterations = 200
    max_iterations = 2000

    percent_labeled = 30

    prior = None

    def __init__(self):
        pass

    def learn(self, graph, nodes, relational):
        prior = relational.learn(graph, nodes)

        if prior:
            self.prior = prior
        else:
            # Need this for wvRN (prior is not learned)
            self.prior = {'+': 0.5, '-': 0.5}
            relational.prior = self.prior['+']

    def predict(self, graph, nodes, relational):
        burn_in = 0
        iterations = 0

        # Step 1: Initialization  - - - - - - *
        # Init V^U with M_L -- don't have an M_L right now; use priors
        tosses = [random.uniform(0, 1) for o in range(len(nodes))]
        
        # TODO: Get rid of this (only for 1 experiment)
        # tosses_2 = [random.uniform(0, 1) for o in range(len(nodes))]
        # assignments = [[nodes[o]['class']] if tosses_2[i] <= 0.8 else ['+'] if t <= int(self.prior['+']) else ['-']
        #                for i, t in enumerate(tosses)]

        assignments = [['+'] if t <= int(self.prior['+']) else ['-']
                       for t in tosses]

        nodes['init'] = assignments
        nodes['assignments'] = assignments

        # Step 2: Random Ordering - - - - - - *
        ordering = range(len(nodes)) # will be less than nodes in entire graph
        random.shuffle(ordering)

        # Step 2.5: % labels bootstrap  - - - *
        if self.percent_labeled > 0:
            # We will bootstrap from the nodes in the training set
            if 'id' not in nodes.attributes():
                graph.vs['id'] = range(len(graph.vs))
            labeling = graph.vs.select(partition_eq=0)['id']
            random.shuffle(labeling)
            upto = len(nodes) * (self.percent_labeled / 100)
            to_label = labeling[:upto + 1]

            graph.vs.select(to_label)['use_label'] = True

        # Step 3: Iterate - - - - - - - - - - *
        while (burn_in <= self.burn_in_iterations and iterations < self.max_iterations):
            for o in ordering:
                prob_pos, prob_neg = relational.predict(graph, nodes[o])

                if np.isfinite(prob_pos) and np.isfinite(prob_neg):
                    if (prob_pos + prob_neg) < 0.9:
                        # Scale (pseudo?) probabilities for coin flip
                        prob_total = float(prob_pos + prob_neg)
                        prob_pos = prob_pos / prob_total
                        prob_neg = prob_neg / prob_total

                # Flip coin
                toss = random.uniform(0, 1)

                if np.isfinite(prob_pos):
                    if toss <= prob_pos:
                        label = '+'
                    else:
                        label = '-'
                elif np.isfinite(prob_neg):
                    if toss > prob_neg:
                        label = '-'
                    else:
                        label = '+'
                else:
                    if toss <= self.prior['+']:
                        label = '+'
                    else:
                        label = '-'

                nodes[o]['assignments'].append(label)
                nodes[o]['pred'] = label

            if burn_in < self.burn_in_iterations:
                burn_in += 1
                # print 'Burnin %s done' % burnin
            else:
                iterations += 1
                # print 'Iteration %s done' % iters


        for n in nodes:
            assigned = n['assignments'][burn_in:]
            n['probability'] = assigned.count('+') / float(len(assigned))

        # Note: Probabilities are stored in graph

class TestUpperBound():
    '''Collective Inference Method: None (Use classifier predictions)'''

    prior = None

    def __init__(self):
        pass

    def learn(self, graph, nodes, relational):
        relational.test = True

        prior = relational.learn(graph, nodes)

        if prior:
            self.prior = prior
        else:
            # Need this for wvRN (prior is not learned)
            self.prior = {'+': 0.5, '-': 0.5}
            relational.prior = self.prior['+']

    def predict(self, graph, nodes, relational):
        # Step 2: Random Ordering - - - - - - *
        ordering = range(len(nodes)) # will be less than nodes in entire graph
        random.shuffle(ordering)

        # Step 3: Iterate - - - - - - - - - - *
        for o in ordering:
            prob_pos, prob_neg = relational.predict(graph, nodes[o])

            if np.isfinite(prob_pos) and np.isfinite(prob_neg):
                if (prob_pos + prob_neg) < 0.9:
                    # Scale (pseudo?) probabilities for coin flip
                    prob_total = float(prob_pos + prob_neg)
                    prob_pos = prob_pos / prob_total
                    prob_neg = prob_neg / prob_total

            # Do new prediction
            if prob_pos >= prob_neg:
                label = '+'
            else:
                label = '-'

            nodes[o]['probability'] = prob_pos
            nodes[o]['pred'] = label