from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score
import numpy as np

def ARI_score(nodes, train_com, pred_com):
    generated_labels = []
    real_labels = []

    for node in nodes:
        if node in pred_com:
            generated_labels.append(1)
        else:
            generated_labels.append(-1)
        if node in train_com:
            real_labels.append(1)
        else:
            real_labels.append(-1)

    ari_score = adjusted_rand_score(real_labels, generated_labels)

    return ari_score

def f1_score(pre_com, true_com):

    set_a = set(pre_com)
    set_b = set(true_com)

    intersection = set_a.intersection(set_b)
    precision = len(intersection) / len(set_b) if len(set_b) > 0 else 0
    recall = len(intersection) / len(set_a) if len(set_a) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def nmi_score(pre_com, true_com, num_node):
    total_nodes = np.arange(num_node)
    pre = np.array([1 if node in pre_com else 0 for node in total_nodes])
    true = np.array([1 if node in true_com else 0 for node in total_nodes])

    nmi = normalized_mutual_info_score(pre, true)

    return nmi

def jac_score(pre_com, true_com):
    set1 = set(pre_com)
    set2 = set(true_com)

    jac = len(set1.intersection(set2)) / len(set1.union(set2))

    return jac

def matrics(pre_coms, true_coms, num_node):
    f1_sum, nmi_sum, jac_sum = 0, 0, 0

    for i in range(len(true_coms)):
        f1_sum += f1_score(pre_coms[i], true_coms[i])
        nmi_sum += nmi_score(pre_coms[i],  true_coms[i], num_node)
        jac_sum += jac_score(pre_coms[i],  true_coms[i])

    return f1_sum / len(true_coms), nmi_sum / len(true_coms), jac_sum / len(true_coms)

def boxline(pre_coms, true_coms, num_node):
    f1, nmi, jac = [], [], []

    for i in range(len(true_coms)):
        f1.append(f1_score(pre_coms[i], true_coms[i]))
        nmi.append(nmi_score(pre_coms[i],  true_coms[i], num_node))
        jac.append(jac_score(pre_coms[i],  true_coms[i]))

    return f1, nmi, jac