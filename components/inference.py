import numpy as np

    # Variables to fill-in
    # nodes = None  # dict of dicts
    # adjacency = None  # dict of lists

ORACLE = 1
AVERAGE = 2
MAX_MIN = 3
INVERT_MAX_MIN = 4
INTERLEAVING = 5


class Inference:

    def __init__(self, graph, collective, relational):
        '''
        :param collective: the collective inference method to use
        :param relational: the relational classifier to use
        '''
        # print len(graph.vs)
        self.graph = graph.copy()

        # Add node ids
        # if 'id' not in self.graph.vs.attributes():
        self.graph.vs['id'] = range(self.graph.vcount())

        self.collective = collective
        self.relational = relational

    def learn(self, partition):
        # print partition
        # print len(self.graph.vs)
        nodes_p = self.graph.vs(partition_eq=partition)

        # Learning step
        # update = collective.learn(nodes_p, relational)
        # _update_nodes(update)

        self.collective.learn(self.graph, nodes_p, self.relational)

    def predict(self, partition):
        if partition:
            nodes_p = self.graph.vs(partition_eq=partition)
        else:
            nodes_p = self.graph.vs()

        # Prediction step
        # update = collective.predict(nodes_p, relational)
        # _update_nodes(update)

        self.collective.predict(self.graph, nodes_p, self.relational)


class Ensemble:
    # Ensemble of Naive Bayes and Weighted-Voting

    def __init__(self, graph, collective_name, type=ORACLE):
        # Initialization: same as Inference
        self.graph = graph.copy()

        # Add node ids
        # if 'id' not in self.graph.vs.attributes():
        self.graph.vs['id'] = range(self.graph.vcount())

        # Ensemble part
        self.type = type    # save type of ensemble
        from components import relational, collective
        self.col = collective_name
        if self.type == INTERLEAVING:
            self.ensemble = collective.EnsembleInterleaving()

        # Instantiate WV & ICA, NB & (GS OR RL)
        self.WV = {'method': None, 'graph': self.graph.copy()}
        self.WV['method'] = Inference(self.WV['graph'], collective.IterativeClassification(), relational.wvRN())
        self.WV['method'].collective.percent_labeled = 0

        self.NB = {}
        if collective_name == 'GS':
            self.NB[collective_name] = {'method': None, 'graph': self.graph.copy()}
            self.NB[collective_name]['method'] = Inference(self.NB[collective_name]['graph'], collective.RelaxationLabeling(), relational.nBC())
            self.NB[collective_name]['method'].collective.percent_labeled = 0
        elif collective_name == 'RL':
            self.NB[collective_name] = {'method': None, 'graph': self.graph.copy()}
            self.NB[collective_name]['method'] = Inference(self.NB[collective_name]['graph'], collective.GibbsSampling(), relational.nBC())
            self.NB[collective_name]['method'].collective.percent_labeled = 0

    def learn(self, partition):
        self.WV['method'].learn(partition)
        self.NB[self.col]['method'].learn(partition)

    def predict(self, partition):
        if self.type != INTERLEAVING:
            self.WV['method'].predict(partition)
            self.NB[self.col]['method'].predict(partition)

            probs = {'WV': self.WV['method'].graph.vs(partition_eq=partition)['probability'],
                     'NB': self.NB[self.col]['method'].graph.vs(partition_eq=partition)['probability']}

            classes = self.graph.vs(partition_eq=partition)['class']
            nodes_p = self.graph.vs(partition_eq=partition)

            nodes_p['WV'] = probs['WV']
            nodes_p['NB'] = probs['NB']

            for i, v_class in enumerate(classes):
                tmp = []

                if np.isfinite(probs['WV'][i]):
                    tmp.append(probs['WV'][i])

                if np.isfinite(probs['NB'][i]):
                    tmp.append(probs['NB'][i])

                if len(tmp) > 0:
                    if self.type == ORACLE:
                        # Pick estimate closest to truth (i.e., cheat)
                        if v_class == 1:
                            nodes_p[i]['probability'] = max(tmp)
                        else:
                            nodes_p[i]['probability'] = min(tmp)
                    elif self.type == AVERAGE:
                        # Take average estimates
                        nodes_p[i]['probability'] = np.mean(tmp)
                    elif self.type == MAX_MIN:
                        # Pick highest estimate
                        if np.mean(tmp) >= 0.5:
                            nodes_p[i]['probability'] = max(tmp)
                        else:
                            nodes_p[i]['probability'] = min(tmp)
                    elif self.type == INVERT_MAX_MIN:
                        # Use more conservative estimate
                        if np.mean(tmp) >= 0.5:
                            nodes_p[i]['probability'] = min(tmp)
                        else:
                            nodes_p[i]['probability'] = max(tmp)
        else:
            ensemble_data = {'col': self.col, 'WV': self.WV, 'NB': self.NB, 'partition': partition}
            self.graph = self.ensemble.predict(graph=self.graph, data=ensemble_data)
