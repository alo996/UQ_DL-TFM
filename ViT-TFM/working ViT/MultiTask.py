import torch


def append_predictions_and_targets(predictions, targets, device):
    """
    Extract predicted tractions per patch for every sample.

    Parameters
    __________
    predictions: torch.Tensor
        Shape `(n_samples, 2, dspl_size, dspl_size)`

    targets: torch.Tensor
        Shape `(n_samples, 3, dspl_size, dspl_size)`

    Returns
    _______
    torch.Tensor
        Shape `(n_samples, 50, 2, dspl_size, dspl_size)`
    """
    n_samples, _, dspl_size, dspl_size = targets.shape
    appended_predictions = torch.zeros((n_samples, 50, 2, dspl_size, dspl_size))  # (n_samples, 50, 2, dspl_size, dspl_size))
    appended_targets = torch.zeros((n_samples, 50, 2, dspl_size, dspl_size))  # (n_samples, 50, 2, dspl_size, dspl_size))
    targets = torch.repeat_interleave(targets, torch.tensor([1, 1, 2]).to(device), dim=1, output_size=4).to(device)
    #indices = torch.arange(start=1, end=50, step=1)
    #segmented_targets = torch.where(targets[:, 2:, :, :] == indices, targets[:, 0:2, :, :], 0)  # (n_samples, 2, dspl_size, dspl_size)
    #print(segmented_targets.shape)

    for i in range(0, 50):
        segmented_targets = torch.where(targets[:, 2:, :, :] == i + 1, targets[:, 0:2, :, :], 0)  # (n_samples, 2, dspl_size, dspl_size)
        segmented_predictions = torch.where(targets[:, 2:, :, :] == i + 1, predictions[:, 0:2, :, :], 0)  # (n_samples, 2, dspl_size, dspl_size)
        appended_predictions[:, i, :, :, :] = segmented_predictions
        appended_targets[:, i, :, :, :] = segmented_targets


    return appended_predictions, appended_targets


def dtma(appended_predictions, appended_targets, device, per_sample=False):
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
        dspl_size = appended_targets.shape[3]
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(2)
        normalization = 50 - torch.sum(temp, 1)

    l2_pred = torch.linalg.vector_norm(appended_predictions + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)

    l2_real = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2, dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)

    l2_pred = torch.mean(l2_pred, dim=(2, 3))  # shape (batch_size, 50)
    l2_real = torch.mean(l2_real, dim=(2, 3))  # shape (batch_size, 50)

    nominator = (1 / normalization) * torch.sum(l2_pred - l2_real, dim=1).to(device)  # shape (batch_size)
    denominator = (1 / normalization) * torch.sum(l2_real, dim=1).to(device)  # shape (batch_size)

    if per_sample:
        return nominator / (denominator + 1e-07)
    else:
        return torch.mean(nominator / (denominator + 1e-07))


def dda(appended_predictions, appended_targets, device, per_sample=False):
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
        dspl_size = appended_targets.shape[3]
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(2)
        normalization = 50 - torch.sum(temp, 1)

    unit_vecs_preds = torch.nn.functional.normalize(appended_predictions, p=2, dim=2)
    unit_vecs_targets = torch.nn.functional.normalize(appended_targets, p=2, dim=2)

    dot_prod = (unit_vecs_preds[:, :, 0, :, :] * unit_vecs_targets[:, :, 0, :, :]) + (
                unit_vecs_preds[:, :, 1, :, :] * unit_vecs_targets[:, :, 1, :, :])
    with torch.no_grad():
        clamped_dot_prod = torch.clamp(dot_prod, -0.999999, 0.999999)
    angles = torch.where(dot_prod != 0., torch.arccos(clamped_dot_prod), 0.)  # shape (batch_size, 50, dspl_size, dspl_size)
    norm_1 = 1 / (torch.count_nonzero(angles, dim=(2, 3)) + 1e-07).to(device)  # shape (batch_size, 50)
    mean_angle_per_patch = norm_1 * torch.sum(angles, dim=(2, 3)).to(device)  # shape (batch_size, 50)

    dda_per_sample = (1 / normalization) * torch.sum(mean_angle_per_patch, dim=1).to(device)

    if per_sample:
        return dda_per_sample
    else:
        return torch.mean(dda_per_sample)


def multi_task_loss(predictions, labels, device):
    alpha_1 = 1/3
    alpha_2 = 1/3
    alpha_3 = 1/3
    mse = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        appended_predictions, appended_targets = append_predictions_and_targets(predictions, labels, device=device)
    DTMA = dtma(appended_predictions, appended_targets, device)
    DTMA = torch.sqrt(DTMA ** 2)
    DDA = dda(appended_predictions, appended_targets, device)

    return alpha_1 * mse(predictions, labels[:, 0:2, :, :]) + alpha_2 * DTMA + alpha_3 * DDA
