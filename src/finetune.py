import torch


def finetune_epoch(model, data_loader, loss_fn, optimizer, scheduler, meters, epoch, num_epochs, clip_grad_norm=True, print_every=100):
    num_batches = len(data_loader)
    model.train()
    
    for meter in meters:
        meter.reset()

    for i, (data, labels) in enumerate(data_loader):
        data = data.cuda()
        labels = labels.cuda()
        
        step = i + (epoch - 1) * num_batches
        scheduler(step)
        optimizer.zero_grad()

        logits = model(data)
        loss = loss_fn(logits, labels)
        loss.backward()

        if clip_grad_norm:
            for param_group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group["params"], 1.0)

        optimizer.step()
        
        meters['loss'].add(float(loss.item()), data.data.shape[0])
        meters['accuracy'].add(logits, labels, data.data.shape[0])
        
        if step % print_every == 0:
            step_loss = loss.item()
            print(f'Fine-tuning Step Loss: {step_loss:.6f} | Epoch: {epoch}/{num_epochs} | Step: {step}')
            
    return meters['loss'].value(), meters['accuracy'].value()
            
        