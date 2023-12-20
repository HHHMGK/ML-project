import torch

def P_at_K(k, pred, truth):
    # print(pred)
    _, indices = torch.topk(pred, k=k)
    correct = 0
    for id in indices:
        if truth[id] > 0:
            correct += 1
    return correct / k

def AP_at_K(k, pred, truth):
    AP = 0
    for i in range(1, k+1):
        AP += P_at_K(i, pred, truth) 
    return AP / k

def MAP_at_K(k, pred_list, truth_list):
    MAP = 0
    for i in range(len(pred_list)):
        MAP += AP_at_K(k, pred_list[i], truth_list[i])
    return MAP / len(pred_list)


def normalize(pred, topk=False):
    pred_1 = torch.zeros(pred.shape)
    for i in range(pred.shape[0]):
        if topk:
            ids = torch.topk(pred[i], k=3).indices
        else:
            ids = [j*(pred[i][j] > 0.0) for j in range(len(pred[i]))]
        for id in ids:
            pred_1[i][id] = 1
    return pred_1