class Inference():

    # Variables to fill-in
    # nodes = None  # dict of dicts
    # adjacency = None  # dict of lists

    # graph = None

    partition_field = None

    c_label = None
    c_hat_label = None
    p_hat_label = None

    # collective = None
    # relational = None

    def __init__(self, graph, collective, relational):
        '''
        :param collective: the collective inference method to use
        :param relational: the relational classifer to use
        '''
        # print len(graph.vs)
        self.graph = graph.copy()

        # Add node ids
        # if 'id' not in self.graph.vs.attributes():
        self.graph.vs['id'] = range(self.graph.vcount())

        self.collective = collective
        self.relational = relational

    # def _update_nodes(self, update):
    #     for u in update:
    #         self.nodes[u[nid]] = u

    def learn(self, partition):
        # print partition
        # print len(self.graph.vs)
        nodes_p = self.graph.vs(partition_eq=partition)

        # Learning step
        # update = collective.learn(nodes_p, relational)
        # _update_nodes(update)

        self.collective.learn(self.graph, nodes_p, self.relational)

    def predict(self, partition):
        nodes_p = self.graph.vs(partition_eq=partition)

        # Prediction step
        # update = collective.predict(nodes_p, relational)
        # _update_nodes(update)

        self.collective.predict(self.graph, nodes_p, self.relational)
