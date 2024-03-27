import torch
import tqdm


def finetune_epoch(model, data_loader, loss_fn, optimizer, scheduler, meter, device, epoch, num_epochs, clip_grad_norm=True, print_every=100):
    num_batches = len(data_loader)
    model.train()
    meter.reset()

    for i, (data, labels) in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        labels = labels.to(device)
        
        step = i + epoch * num_batches
        scheduler(step)
        optimizer.zero_grad()

        logits = model(data)
        loss = loss_fn(logits, labels)
        meter.add(float(loss.item()), data.data.shape[0])

        loss.backward()

        if clip_grad_norm:
            for param_group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(param_group["params"], 1.0) # TODO: check if this is necessary

        optimizer.step()

        if step % print_every == 0:
            step_loss = loss.item() / data.data.shape[0]
            print(f'Fine-tuning Step Loss: {step_loss:.6f} | Epoch: {epoch}/{num_epochs} | Step: {step}')
            
    return meter.value()
            
        