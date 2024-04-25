import torch
import numpy as np
from scipy.sparse.linalg import svds


def compute_neural_collapse(image_encoder, data_loader, num_classes):
    image_encoder.eval()
    num_resblocks = len(image_encoder.transformer.resblocks)
    Sw_invSb = []

    for l in range(1, num_resblocks + 1):
        mean = [0 for _ in range(num_classes)]
        Sw = 0
        examples_in_class = [0 for _ in range(num_classes)]

        for computation in ['Mean','Cov']:
            for _, (data, labels) in enumerate(data_loader, start=1):
                data = data.cuda() # (B, 3, 224, 224)

                # feed data through model up until transformer
                x = image_encoder.conv1(data) # (B, 768, S, S) where S = 224 / patchsize = 16
                x = x.reshape(x.shape[0], x.shape[1], -1) # (B, 768, S') where S' = S^2 = 192
                x = x.permute(0, 2, 1) # (B, S', 768)

                x = torch.cat([image_encoder.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1) # (B, 197, 768)
                x = x + image_encoder.positional_embedding.to(x.dtype) # (B, S'', 768) where S'' = S' + 1

                x = image_encoder.patch_dropout(x)
                x = image_encoder.ln_pre(x) # (B, S'', 768)
                x = x.permute(1, 0, 2)  # (S'', B, 768)  NLD -> LND

                # get features at resblock layer l
                for r in image_encoder.transformer.resblocks[:l]:
                    x = r(x)
                h = x[0].detach().cpu() # (B, 768)

                # compute mean and covariance
                for c in range(num_classes):
                    if computation == 'Mean':
                        examples_in_class[c] += labels[labels == c].shape[0]
                        mean[c] += h[labels == c].sum(dim=0)

                    elif computation == 'Cov':
                        q = h[labels == c] - mean[c] # q_c = h_c - mu_c
                        q = q.unsqueeze(dim=-1) # (B, 768, 1)
                        Sw += torch.bmm(q, q.transpose(dim0=-2, dim1=-1)).sum(dim=0) # (768, 768)

            # normalize the sum by the number of contributions
            if computation == 'Mean':
                for c in range(num_classes):
                    mean[c] /= examples_in_class[c]
                    M = torch.stack(mean).T # (768, C)
            elif computation == 'Cov':
                Sw /= sum(examples_in_class) # (768, 768)

        # global mean
        muG = torch.mean(M, dim=1, keepdim=True) # (768, 1)

        # between-class covariance
        M_ = M - muG # (768, num classes)
        Sb = torch.matmul(M_, M_.T) / num_classes # (768, 768)

        # NC1: activation collapse
        Sw = Sw.numpy() # (768, 768)
        Sb = Sb.numpy() # (768, 768)
        eigvec, eigval, _ = svds(Sb, k=num_classes-1)
        inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T
        Sw_invSb.append(np.trace(Sw @ inv_Sb) / num_classes)

    return Sw_invSb