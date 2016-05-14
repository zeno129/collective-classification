import random


def Sampling_Random(graph, percent_label=50):
    nodes = graph.vs

    labeling = range(len(nodes))
    random.shuffle(labeling)
    upto = len(nodes) * (percent_label / 100)
    to_label = labeling[:upto + 1]

    for l in to_label:
        nodes[l]['use_label'] = True
        nodes[l]['partition'] = 0

    for n in nodes:
        if n['partition'] != 0:
            n['partition'] = 1

    return graph


def Sampling_IC(graph, num_partitions=2):
    '''
    Sampling as described in:

    Neville, Jennifer, and David Jensen. "Iterative classification in relational data."
    Proc. AAAI-2000 Workshop on Learning Statistical Models from Relational Data. 2000.

    '''
    # 1. Initialize X to the set of all company objects.
    nodes = graph.vs
    ordering = range(len(nodes))
    random.shuffle(ordering)
    # print ordering

    partitions = range(num_partitions)
    partition_assignments = [False] * num_partitions

    discarded = []

    # 2. Do until X is empty:
    while ordering:
        for p in partitions:
            if not partition_assignments[p] and ordering:
                # while something
                while not partition_assignments[p] and ordering:
                    # i. Randomly pick a node x
                    o = ordering.pop()
                    x = nodes[o]
                    # ii. Gather all nodes one link away
                    neighbors = nodes.select(graph.neighborhood(x))

                    # Reset
                    skip = True

                    # iii. Check if any nodes is in different sample
                    if 'partition' in neighbors.attributes():
                        counts = neighbors['partition'].count(p) + neighbors['partition'].count(None)
                        if counts == len(neighbors):
                            skip = False
                    else:
                        skip = False

                    if skip:
                        # Leave for later
                        discarded.append(o)
                    else:
                        # Assign to partition
                        partition_assignments[p] = True
                        x['partition'] = p
                        neighbors['partition'] = [p] * len(neighbors)

        # Reset stuff
        partition_assignments = [False] * num_partitions

    # 3. For all discarded companies, randomly place in partitions
    random.shuffle(discarded)
    slices = int(len(discarded) / num_partitions)
    remainder = len(discarded) % num_partitions

    for p in partitions:
        start = p * slices
        end = start + slices
        tmp = discarded[start:end]
        nodes.select(tmp)['partition'] = p

    # 4. Label all nodes with no links to other partitions as 'core'
    for n in nodes:
        p = n['partition']
        neighbors = nodes.select(graph.neighborhood(n))
        if len(neighbors) == neighbors['partition'].count(p):
            n['core'] = True

    return graph
        

