    # pos=nx.spring_layout(SG)
    # pos=nx.circular_layout(SG)
    # pos=graphviz_layout(SG, prog='dot')
    # pos=nx.spectral_layout(SG)


    # Only including certain weight will favor high-degree nodes
    # MIN_WEIGHT = 10 * 1000
    # MIN_DEGREE = 1000 * 1000
    # MAX_DEGREE = 3000 * 1000

    # nodes = [node for node, degree in SG.degree(weight='weight').items() if degree > MIN_DEGREE]
    # nodes = [node for node, degree in SG.degree(weight='weight').items() if degree < MAX_DEGREE and node in nodes]
    # SG = SG.subgraph(nodes)
