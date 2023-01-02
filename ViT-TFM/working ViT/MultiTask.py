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


def dtma(appended_predictions, targets, device, per_sample=False):
    """
    Calculates the DTMA as described by Huan et al. (2019).

    Parameters
    __________
    appended_predictions: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    device: torch.device
        Device (e.g. cpu or gpu)

    per_sample : Bool
        Whether to return the dtma for each sample rather than the average.

    Returns
    _______
    float
        DTMA
    """
    with torch.no_grad():
        dspl_size = targets.shape[3]
        temp = (targets[:, :] == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(2)
        normalization = 50 - torch.sum(temp, 1)

    l2_pred = torch.linalg.vector_norm(appended_predictions + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)

    l2_real = torch.linalg.vector_norm(targets + 1e-07, ord=2, dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)

    l2_pred = torch.mean(l2_pred, dim=(2, 3))  # shape (batch_size, 50)
    l2_real = torch.mean(l2_real, dim=(2, 3))  # shape (batch_size, 50)

    nominator = (1 / normalization) * torch.sum(l2_pred - l2_real, dim=1)  # shape (batch_size)
    denominator = (1 / normalization) * torch.sum(l2_real, dim=1)  # shape (batch_size)

    if per_sample:
        return nominator / (denominator + 1e-07)
    else:
        return torch.mean(nominator / (denominator + 1e-07))


def dda(appended_predictions, targets, device, per_sample=False):
    """
    Calculates the DDA as described by Kierfeld et al. (2022).

    Parameters
    __________
    appended_predictions: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(batch_size, 50, 2, dspl_size, dspl_size)`

    device: torch.device
        Device (e.g. cpu or gpu)

    per_sample : Bool
        Whether to return the dtma for each sample rather than the average.

    Returns
    _______
    float
        DDA
    """
    with torch.no_grad():
        dspl_size = targets.shape[3]
        temp = (targets[:, :] == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(2)
        normalization = 50 - torch.sum(temp, 1)

    unit_vecs_preds = torch.nn.functional.normalize(appended_predictions, p=2, dim=2)
    unit_vecs_targets = torch.nn.functional.normalize(targets, p=2, dim=2)

    dot_prod = (unit_vecs_preds[:, :, 0, :, :] * unit_vecs_targets[:, :, 0, :, :]) + (
                unit_vecs_preds[:, :, 1, :, :] * unit_vecs_targets[:, :, 1, :, :])
    with torch.no_grad():
        clamped_dot_prod = torch.clamp(dot_prod, -0.999999, 0.999999)
    angles = torch.where(dot_prod != 0., torch.arccos(clamped_dot_prod), 0.)  # shape (batch_size, 50, dspl_size, dspl_size)
    norm_1 = 1 / (torch.count_nonzero(angles, dim=(2, 3)) + 1e-07)  # shape (batch_size, 50)
    mean_angle_per_patch = norm_1 * torch.sum(angles, dim=(2, 3))  # shape (batch_size, 50)

    dda_per_sample = (1 / normalization) * torch.sum(mean_angle_per_patch, dim=1)

    if per_sample:
        return dda_per_sample
    else:
        return torch.mean(dda_per_sample)


def multi_task_loss(predictions, appended_labels, device):
    labels = torch.sum(appended_labels, dim=1)
    alpha_1 = 1/3
    alpha_2 = 1/3
    alpha_3 = 1/3
    mse = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        appended_predictions = append_predictions(predictions, appended_labels)
    DTMA = dtma(appended_predictions, appended_labels, device)
    DTMA = torch.sqrt(DTMA ** 2)
    DDA = dda(appended_predictions, appended_labels, device)
    return alpha_1 * mse(predictions, labels) + alpha_2 * DTMA + alpha_3 * DDA
