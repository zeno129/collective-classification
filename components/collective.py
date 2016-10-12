import random
import numpy as np
import math

def inv_sigmoid(x):
    return np.log(x / float((1 - x)))

def sigmoid(x):
    return 1.0 / float((1.0 + np.exp(-x)))

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
                    # graph.vs['id'] = range(graph.vcount())
                    labeling = graph.vs.select(partition_eq=0)['id']
                    random.shuffle(labeling)
                    upto = int(math.ceil(len(nodes) * (self.percent_labeled / float(100))))
                    to_label = labeling[:upto + 1]

                    graph.vs.select(to_label)['use_label'] = True

                # Step 3: Iterate - - - - - - - - - - *
                # Store probabilities in a list (same order)
                prob_pos_iter = [relational.predict(graph, nodes[o])[0] for o in ordering]

                # Assign probabilities to all at same time
                # nodes[ordering]['probability'] = probs_iter
                # nodes[ordering]['pred'] = labels_iter

                # Do Joel's probability correction - - - - - - - - *
                # Step 1: transform probabilities to logit space
                sig_probs_iter = [inv_sigmoid(p) if (np.isfinite(p) and p < 1) else p for p in prob_pos_iter]

                # Step 2: calculate offset location
                tmp = sorted(sig_probs_iter)
                phi = self.prior['-'] * len(prob_pos_iter)
                h_phi = tmp[int(phi)]

                # Step 3: adjust logits
                sig_probs_iter = [h_i - h_phi for h_i in sig_probs_iter]

                # Step 4: transform back to probabilities
                prob_pos_iter = [sigmoid(h_i) for h_i in sig_probs_iter]

                # - - - - - - - - - - - - - - - - - - - - - - - - *

                nodes[ordering]['probability'] = prob_pos_iter
                nodes[ordering]['pred'] = ['+' if p >= 0.5 else '-' for p in prob_pos_iter]


                if 'pred' in nodes.attributes() and nodes['pred'] == prev_labels:
                    classes_changed = False

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
            # graph.vs['id'] = range(graph.vcount())
            labeling = graph.vs.select(partition_eq=0)['id']
            random.shuffle(labeling)
            upto = int(math.ceil(len(nodes) * (self.percent_labeled / float(100))))
            to_label = labeling[:upto + 1]

            graph.vs.select(to_label)['use_label'] = True

        # Step 3: Iterate - - - - - - - - - - *
        while (burn_in <= self.burn_in_iterations and iterations < self.max_iterations):
            # Store probabilities in a list (same order)
            prob_pos_iter = [relational.predict(graph, nodes[o])[0] for o in ordering]

            # Do Joel's probability correction - - - - - - - - *
            # Step 1: transform probabilities to logit space
            sig_probs_iter = [inv_sigmoid(p) if (np.isfinite(p) and p < 1) else p for p in prob_pos_iter]

            # Step 2: calculate offset location
            tmp = sorted(sig_probs_iter)
            phi = self.prior['-'] * len(prob_pos_iter)
            h_phi = tmp[int(phi)]

            # Step 3: adjust logits
            sig_probs_iter = [h_i - h_phi for h_i in sig_probs_iter]

            # Step 4: transform back to probabilities
            prob_pos_iter = [sigmoid(h_i) for h_i in sig_probs_iter]

            # - - - - - - - - - - - - - - - - - - - - - - - - *

            for i, o in enumerate(ordering):
                prob_pos = prob_pos_iter[i]

                # Flip coin
                toss = random.uniform(0, 1)

                if np.isfinite(prob_pos):
                    if toss <= prob_pos:
                        label = '+'
                    else:
                        label = '-'
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
    '''Collective Inference Method: None (Use classifier predictions; i.e., oracle)'''

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

        # graph.vs['id'] = range(graph.vcount())

        # Step 3: Iterate - - - - - - - - - - *
        for o in ordering:
            prob_pos, prob_neg = relational.predict(graph, nodes[o])

            # Do new prediction
            if prob_pos >= prob_neg:
                label = '+'
            else:
                label = '-'

            nodes[o]['probability'] = prob_pos
            nodes[o]['pred'] = label
