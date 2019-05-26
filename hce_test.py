import networkx as nx
import matplotlib.pyplot as plt
from hce import hce


if __name__ == '__main__':
    graphs = [
        nx.karate_club_graph(),
        nx.davis_southern_women_graph(),
        nx.les_miserables_graph(),
        nx.florentine_families_graph(),
        nx.powerlaw_cluster_graph(500, 1, 0.001),
    ]
    labels = [
        "Karate",
        "Davis Southern Women",
        "Les Miserables",
        "Florentine Families",
        "Power Law Cluster",
    ]

    for G, l in zip(graphs, labels):
        fig, ax = plt.subplots()
        nx.draw(
            G,
            pos=hce(G),
            ax=ax,
        )
        plt.savefig(f"{l.lower().replace(' ', '_')}_test.png")
        plt.close()
