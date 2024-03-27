import torch
import tqdm 


def evaluate(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        correct, n = 0., 0.
        for _, (data, labels) in enumerate(tqdm.tqdm(data_loader)):
            # maybe dictionarize here? reference task arithmetic code 
            data = data.to(device)
            labels = labels.to(device)
            
            logits = model(data)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

        accuracy = correct / n # top1
    
    return accuracy