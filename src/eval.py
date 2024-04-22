import torch


def evaluate(model, data_loader):
    model.eval()

    with torch.no_grad():
        correct, n = 0., 0.
        for _, (data, labels) in enumerate(data_loader):
            # maybe dictionarize here? reference task arithmetic code 
            data = data.cuda()
            labels = labels.cuda()
            
            logits = model(data)
            pred = logits.argmax(dim=1, keepdim=True)

            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

        accuracy = correct / n # top1
    
    return accuracy