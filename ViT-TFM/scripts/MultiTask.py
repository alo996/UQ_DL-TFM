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

    for i in range(0, 50):
        segmented_targets = torch.where(targets[:, 2:, :, :] == i + 1, targets[:, 0:2, :, :], 0)  # (n_samples, 2, dspl_size, dspl_size)
        segmented_predictions = torch.where(targets[:, 2:, :, :] == i + 1, predictions[:, 0:2, :, :], 0)  # (n_samples, 2, dspl_size, dspl_size)
        appended_predictions[:, i, :, :, :] = segmented_predictions
        appended_targets[:, i, :, :, :] = segmented_targets

    return appended_predictions, appended_targets


def dtmb(predictions, targets, appended_predictions, appended_targets, device, per_sample=False):
    n_samples, _, dspl_size, dspl_size = targets.shape
    with torch.no_grad():
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    background_targets = torch.zeros((n_samples, 2, dspl_size, dspl_size)).to(device)
    background_predictions = torch.zeros((n_samples, 2, dspl_size, dspl_size)).to(device)
    background_targets[:, 0, :, :] = torch.where(targets[:, 2, :, :] > 0, background_targets[:, 0, :, :],
                                                 targets[:, 0, :, :])  # (n_samples, 2, dspl_size, dspl_size)
    background_targets[:, 1, :, :] = torch.where(targets[:, 2, :, :] > 0, background_targets[:, 1, :, :],
                                                 targets[:, 1, :, :])  # (n_samples, 2, dspl_size, dspl_size)
    background_predictions[:, 0, :, :] = torch.where(targets[:, 2, :, :] > 0, background_predictions[:, 0, :, :],
                                                     predictions[:, 0, :, :])  # (n_samples, 2, dspl_size, dspl_size)
    background_predictions[:, 1, :, :] = torch.where(targets[:, 2, :, :] > 0, background_predictions[:, 1, :, :],
                                                     predictions[:, 1, :, :])  # (n_samples, 2, dspl_size, dspl_size)

    num_background_vecs = dspl_size ** 2 - torch.count_nonzero(targets[:, 2, :, :], dim=(1, 2))
    l2_real = torch.linalg.vector_norm(background_targets + 1e-07, ord=2,
                                       dim=1)  # shape (batch_size, dspl_size, dspl_size)
    l2_pred = torch.linalg.vector_norm(background_predictions + 1e-07, ord=2,
                                       dim=1)  # shape (batch_size, dspl_size, dspl_size)
    nominator = (1 / num_background_vecs) * torch.sum(l2_pred - l2_real, dim=(1, 2))  # shape (batch_size)

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    l2_real = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    l2_real = torch.sum(l2_real, dim=(2, 3))
    norm_const = torch.nan_to_num(1 / (num_trac_vecs_per_patch), posinf=0)
    l2_real = norm_const * l2_real  # shape (batch_size, 50)
    denominator = (1 / normalization) * torch.sum(l2_real, dim=1).to(device)  # shape (batch_size)

    if per_sample:
        return nominator / (denominator + 1e-07)
    else:
        return torch.mean(nominator / (denominator + 1e-07))


def snr(predictions, targets, appended_predictions, appended_targets, device, per_sample=False):
    n_samples, _, dspl_size, dspl_size = targets.shape
    with torch.no_grad():
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    l2_pred = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    l2_pred = torch.sum(l2_pred, dim=(2, 3))
    norm_const = torch.nan_to_num(1 / (num_trac_vecs_per_patch), posinf=0)
    l2_pred = norm_const * l2_pred  # shape (batch_size, 50)
    nominator = (1 / normalization) * torch.sum(l2_pred, dim=1).to(device)  # shape (batch_size)

    background_predictions = torch.zeros((n_samples, 2, dspl_size, dspl_size)).to(device)
    background_predictions[:, 0, :, :] = torch.where(targets[:, 2, :, :] > 0, background_predictions[:, 0, :, :],
                                                     predictions[:, 0, :, :])  # (n_samples, 2, dspl_size, dspl_size)
    background_predictions[:, 1, :, :] = torch.where(targets[:, 2, :, :] > 0, background_predictions[:, 1, :, :],
                                                     predictions[:, 1, :, :])  # (n_samples, 2, dspl_size, dspl_size)

    denominator = torch.std(background_predictions)

    if per_sample:
        return nominator / (denominator + 1e-07)
    else:
        return torch.mean(nominator / (denominator + 1e-07))


def dma(predictions, targets, appended_predictions, appended_targets, device, per_sample=False):
    n_samples, _, dspl_size, dspl_size = targets.shape
    with torch.no_grad():
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    l2_pred = torch.linalg.vector_norm(appended_predictions + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    max_norm_per_pred_patch = torch.amax(l2_pred, dim=(2, 3))  # shape(batch_size, 50)
    l2_real = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    max_norm_per_real_patch = torch.amax(l2_real, dim=(2, 3))  # shape(batch_size, 50)
    nominator = max_norm_per_pred_patch - max_norm_per_real_patch

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    norm_const = torch.nan_to_num(1 / (num_trac_vecs_per_patch), posinf=0)
    denominator = norm_const * torch.sum(l2_real, dim=(2, 3))

    average_dma_per_sample = (1 / normalization) * torch.sum(nominator / (denominator + 1e-07), dim=1)

    if per_sample:
        return average_dma_per_sample
    else:
        return torch.mean(average_dma_per_sample)


def dtma(appended_predictions, appended_targets, device, per_sample=False):
    with torch.no_grad():
        dspl_size = appended_targets.shape[3]
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    l2_pred = torch.linalg.vector_norm(appended_predictions + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    l2_real = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    norm_const = torch.nan_to_num(1 / num_trac_vecs_per_patch, nan=0, posinf=0)
    nominator = norm_const * torch.sum(l2_pred - l2_real, dim=(2, 3)).to(device)  # shape (batch_size, 50)
    denominator = norm_const * torch.sum(l2_real, dim=(2, 3)).to(device)  # shape (batch_size, 50)
    average_dtma_per_sample = (1 / normalization) * torch.sum(nominator / (denominator + 1e-07), dim=1)

    if per_sample:
        return average_dtma_per_sample
    else:
        return torch.mean(average_dtma_per_sample)


def adtma(appended_predictions, appended_targets, device, per_sample=False):
    with torch.no_grad():
        dspl_size = appended_targets.shape[3]
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    l2_pred = torch.linalg.vector_norm(appended_predictions + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    l2_real = torch.linalg.vector_norm(appended_targets + 1e-07, ord=2,
                                       dim=2)  # shape (batch_size, 50, dspl_size, dspl_size)
    norm_const = torch.nan_to_num(1 / (num_trac_vecs_per_patch), nan=0, posinf=0)  # shape (batch_size, 50)
    nominator = norm_const * torch.sum(l2_pred - l2_real, dim=(2, 3)).to(device)  # shape (batch_size, 50)
    denominator = norm_const * torch.sum(l2_real, dim=(2, 3)).to(device)  # shape (batch_size, 50)

    average_adtma_per_sample = (1 / normalization) * torch.abs(torch.sum(nominator / (denominator + 1e-07), dim=1))

    if per_sample:
        return average_adtma_per_sample
    else:
        return torch.mean(average_adtma_per_sample)


def cosine_sim(appended_predictions, appended_targets, device, per_sample=False):
    with torch.no_grad():
        dspl_size = appended_targets.shape[3]
        temp = (appended_targets[:, :].to(device) == torch.zeros((2, dspl_size, dspl_size)).to(device)).flatten(2).all(
            2)
        normalization = 50 - torch.sum(temp, 1)

    num_trac_vecs_per_patch = torch.count_nonzero(
        torch.linalg.vector_norm(appended_targets[:, :, 0:2, :, :], ord=2, dim=2), dim=(2, 3))
    cos_theta = torch.nn.functional.cosine_similarity(appended_predictions, appended_targets, dim=2,
                                                      eps=1e-08)  # shape(batch_size, 50, dspl_size, dspl_size)
    norm_const = torch.nan_to_num(1 / (num_trac_vecs_per_patch), nan=0, posinf=0)  # shape (batch_size, 50)
    cos_sim_per_patch = norm_const * torch.sum(cos_theta, dim=(2, 3))
    cos_sim_per_dspl = (1 / normalization) * torch.sum(cos_sim_per_patch, dim=1)

    if per_sample:
        return cos_sim_per_dspl
    else:
        return torch.mean(cos_sim_per_dspl)


def dtma_for_train(appended_predictions, appended_targets, device, per_sample=False):
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


def dda_for_train(appended_predictions, appended_targets, device, per_sample=False):
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
    DDA = dda_for_train(appended_predictions, appended_targets, device)

    return alpha_1 * mse(predictions, labels[:, 0:2, :, :]) + alpha_2 * DTMA + alpha_3 * DDA
