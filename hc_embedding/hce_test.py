import matplotlib.pyplot as plt
import networkx as nx

from hc_embedding import draw_hce


if __name__ == '__main__':
    graphs = [
        nx.karate_club_graph(),
        nx.davis_southern_women_graph(),
        nx.les_miserables_graph(),
        nx.florentine_families_graph(),
        nx.powerlaw_cluster_graph(250, 1, 0.001),
    ]
    labels = [
        "Karate",
        "Davis Southern Women",
        "Les Miserables",
        "Florentine Families",
        "Power Law Cluster",
    ]

    for G, title in zip(graphs, labels):
        draw_hce(G, title)
        plt.savefig(f"{title.lower().replace(' ', '_')}_test.png")
        plt.close()
