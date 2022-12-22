import torch

def append_predictions(predictions, targets):
    """
    Extract predicted tractions per patch for every sample.

    Parameters
    __________
    predictions: torch.Tensor
        Shape `(n_samples, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(n_samples, 50, 2, dspl_size, dspl_size)`

    Returns
    _______
    torch.Tensor
        Shape `(n_samples, 50, 2, dspl_size, dspl_size)`
    """
    n_samples, n_patches, n_channels, trac_size, _ = targets.shape
    binary_masks = torch.where(targets[:, :, :, :, :] != 0, 1, 0)
    appended_predictions = predictions.unsqueeze(1)
    appended_predictions = appended_predictions * binary_masks

    return appended_predictions


def dtma(appended_predictions, targets, device):
    """
    Calculates the DTMA as described by Huan et al. (2019).

    Parameters
    __________
    appended_predictions: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    Returns
    _______
    float
        DTMA
    """
    dspl_size = targets.shape[3]
    temp_1 = torch.zeros((2, dspl_size, dspl_size)).to(device)
    temp_1.requires_grad = False
    temp = (targets[:, :] == temp_1).flatten(2).all((2))
    normalization = 50 - torch.sum(temp, 1)
    print(normalization)

    l2_pred = torch.linalg.vector_norm(appended_predictions, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    l2_real = torch.linalg.vector_norm(targets, ord=2, dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)

    l2_pred = torch.mean(l2_pred, dim=(2, 3))  # shape (batch_size, 50)
    l2_real = torch.mean(l2_real, dim=(2, 3))  # shape (batch_size, 50)

    nominator = (1 / normalization) * torch.sum(l2_pred - l2_real, dim=1)  # shape (batch_size)
    denominator = (1 / normalization) * torch.sum(l2_real, dim=1)  # shape (batch_size)

    dtma = torch.mean(nominator / (denominator + 1e-06))

    return dtma


def dda(appended_predictions, targets, device):
    """
    Calculates the DDA as described by Kierfeld et al. (2022).

    Parameters
    __________
    appended_predictions: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    Returns
    _______
    float
        DDA
    """
    dspl_size = targets.shape[3]
    temp_1 = torch.zeros((2, dspl_size, dspl_size)).to(device)
    temp_1.requires_grad = False
    temp = (targets[:, :] == temp_1).flatten(2).all((2))
    normalization = 50 - torch.sum(temp, 1)
    print(normalization)

    unit_vecs_preds = appended_predictions / (1e-06 + torch.linalg.vector_norm(appended_predictions, ord=2, dim=2,
                                                                               keepdim=True))  # shape (batch_size, 50, 2, dspl_size, dspl_size)
    unit_vecs_targets = targets / (1e-06 + torch.linalg.vector_norm(targets, ord=2, dim=2,
                                                                    keepdim=True))  # shape (batch_size, 50, 2, dspl_size, dspl_size)

    dot_prod = (unit_vecs_preds[:, :, 0, :, :] * unit_vecs_targets[:, :, 0, :, :]) + (
                unit_vecs_preds[:, :, 1, :, :] * unit_vecs_targets[:, :, 1, :, :])
    angles = torch.where(dot_prod != 0., torch.arccos(dot_prod), 0.)  # shape (batch_size, 50, dspl_size, dspl_size)
    norm_1 = 1 / (torch.count_nonzero(angles, dim=(2, 3)) + 1e-06)  # shape (batch_size, 50)
    mean_angle_per_patch = norm_1 * torch.sum(angles, dim=(2, 3))  # shape (batch_size, 50)

    dda = (1 / normalization) * torch.sum(mean_angle_per_patch, dim=1)
    dda = torch.mean(dda)

    return dda


def multi_task_loss(predictions, appended_labels, device):
    labels = torch.sum(appended_labels, dim=1)
    alpha_1 = 1/3
    alpha_2 = 1/3
    alpha_3 = 1/3
    mse = torch.nn.MSELoss(reduction='mean')
    appended_predictions = append_predictions(predictions, appended_labels)
    DTMA = dtma(appended_predictions, appended_labels, device)
    DDA = dda(appended_predictions, appended_labels, device)
    return alpha_1 * mse(predictions, labels) + alpha_2 * DTMA + alpha_3 * DDA
