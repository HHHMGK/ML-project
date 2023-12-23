import torch
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall


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

def print_metrics(pred, truth, thres=0.5):
    print("--------------------------------------")
    print('MAP@1 ', MAP_at_K(1, pred, truth))
    print('MAP@2 ', MAP_at_K(2, pred, truth))
    print('MAP@3 ', MAP_at_K(3, pred, truth))
    print('MAP@4 ', MAP_at_K(4, pred, truth))

    print("--------------------------------------")

    # f1arr = []
    # for thres in range(1, 20):
    #     thres /= 20
    #     f1ma = MultilabelF1Score(num_labels=18, threshold=thres, average='macro')
    #     # f1arr.append((thres,f1ma(pred, truth).tolist()))
    #     print(thres, ' : ', f1ma(pred, truth).tolist())

    # print('mF1 - micro:    ', multilabel_f1_score(pred, truth, num_labels=18, threshold=thres, average='micro').tolist())
    # print('mF1 - macro:    ', multilabel_f1_score(pred, truth, num_labels=18, threshold=thres, average='macro').tolist())
    # print('mF1 - weighted: ', multilabel_f1_score(pred, truth, num_labels=18, threshold=thres, average='weighted').tolist())

    f1ma = MultilabelF1Score(num_labels=18, threshold=thres, average='macro')
    print('F1 macro :', f1ma(pred, truth).tolist())
    f1mi = MultilabelF1Score(num_labels=18, threshold=thres, average='micro')
    print('F1 micro :', f1mi(pred, truth).tolist())

    print("--------------------------------------")
    # from torchmetrics.functional.classification import multilabel_accuracy
    acc = MultilabelAccuracy(num_labels=18, threshold=thres)
    print('Accuracy :', acc(pred, truth).tolist())

    print("--------------------------------------")
    # from torchmetrics.functional.classification import multilabel_precision
    prec = MultilabelPrecision(num_labels=18, threshold=thres, average='macro')
    print('Precision :', prec(pred, truth).tolist())

    print("--------------------------------------")
    # from torchmetrics.functional.classification import multilabel_recall
    rec = MultilabelRecall(num_labels=18, threshold=thres, average='macro')
    print('Recall :', rec(pred, truth).tolist())

def get_metrics(pred, truth, thres=0.5, device='cpu'):
    f1ma = MultilabelF1Score(num_labels=18, threshold=thres, average='macro').to(device)
    f1mi = MultilabelF1Score(num_labels=18, threshold=thres, average='micro').to(device)
    acc = MultilabelAccuracy(num_labels=18, threshold=thres).to(device)
    prec = MultilabelPrecision(num_labels=18, threshold=thres, average='macro').to(device)
    rec = MultilabelRecall(num_labels=18, threshold=thres, average='macro').to(device)

    return f1ma(pred, truth).tolist(), f1mi(pred, truth).tolist(), acc(pred, truth).tolist(), prec(pred, truth).tolist(), rec(pred, truth).tolist()