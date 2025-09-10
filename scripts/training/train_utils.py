import os
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from collections import defaultdict
from loguru import logger

from plume_hunter.dataset import Task
from plume_hunter.dataset import data_stats
from plume_hunter.constants import ROOT_DIR

DEFAULT_VAL = -9999


def update_metrics_classification(
    outputs, targets, accuracy, precision, recall, f1_score
) -> None:
    accuracy.update(outputs, targets)
    precision.update(outputs, targets)
    recall.update(outputs, targets)
    f1_score.update(outputs, targets)


def compute_metrics_classification(
    accuracy, precision, recall, f1_score
) -> list[float]:
    acc = accuracy.compute()
    prec = precision.compute()
    rec = recall.compute()
    f1 = f1_score.compute()

    return acc, prec, rec, f1


def reset_metrics_classification(
    accuracy, precision, recall, f1_score
) -> None:
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()


def update_metrics_sem_segm(outputs, targets, mean_iou) -> None:
    mean_iou.update(outputs.int(), targets.int())


def compute_metrics_sem_segm(mean_iou) -> list[float]:
    res_miou = mean_iou.compute()

    return res_miou


def reset_metrics_sem_segm(mean_iou) -> None:
    mean_iou.reset()


def log_classification(
    logger, wandb, epoch, curr_lr, loss, acc, prec, rec, f1, type_set: str
):
    logger.info(
        f"epoch, current LR, loss, accuracy, precision, recall, f1 score on {type_set} Set: {epoch}, {curr_lr:.8f}, {loss:.6f}, {acc}, {prec}, {rec}, {f1}"
    )

    if wandb.run:
        wandb.log(
            {
                type_set + "_epoch": epoch,
                "current_lr": curr_lr,
                type_set + "_loss": loss,
                type_set + "_accuracy": acc,
                type_set + "_precision": prec,
                type_set + "_recall": rec,
                type_set + "_f1": f1,
            }
        )


def log_sem_segmentation(
    train, logger, wandb, epoch, curr_lr, loss, miou, prec, rec, f1
):
    if train:
        type_set = "Train"
    else:
        type_set = "Val"

    logger.info(
        (
            f"{type_set} Set:\n"
            f"epoch {epoch}, current LR {curr_lr:.12f}, loss {loss:.8f},"
            f" mIoU {miou}, precision {prec}, recall {rec}, f1-score {f1}"
        )
    )
    if train:

        if wandb.run:
            wandb.log(
                {
                    "train_epoch": epoch,
                    "current_lr": curr_lr,
                    "train_loss": loss,
                    "train_mIoU": miou,
                    "train_precision": prec,
                    "train_recall": rec,
                    "train_f1_score": f1,
                }
            )
    else:
        if wandb.run:
            wandb.log(
                {
                    "train_epoch": epoch,
                    "val_loss": loss,
                    "val_mIoU": miou,
                    "val_precision": prec,
                    "val_recall": rec,
                    "val_f1_score": f1,
                }
            )


def log_regression(logger, wandb, epoch, curr_lr, loss, type_set: str):
    logger.info(
        f"epoch, current LR, loss score on {type_set} Set: {epoch}, {curr_lr:.12f}, {loss:.8f}"
    )

    if wandb.run:
        if type_set == "Train":

            wandb.log(
                {
                    "train_epoch": epoch,
                    "current_lr": curr_lr,
                    "train_loss": loss,
                }
            )
        else:
            wandb.log(
                {
                    "val_loss": loss,
                }
            )

    logger.info(f"Total {type_set} loss: {loss}")


def get_model_name_task_timestamp(selected_task: Task) -> str:
    timestamp_training = (
        selected_task.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    return timestamp_training


def get_padding_mask_inputs(imgs: torch.tensor):
    """Gets mask where padding values were added for background.
    This mask will be used to set those values to nan.

    Args:
        imgs (torch.tensor): batch of images N,C,H,W

    Returns:
        _type_: bool mask where True means padding pixels.
    """
    padding_mask = imgs == DEFAULT_VAL
    return padding_mask


def get_no_padding_mask_outs_targets(imgs: torch.tensor):
    """Gets mask of not padded values.
    This mask will be used to filter out which pixels to
    keep when computing the loss for semantic segm and
    concentration regression (not useful for image classification).

    Args:
        imgs (torch.tensor): batch of images N,C,H,W

    Returns:
        _type_: bool mask where True means not padded pixels.
    """
    no_pad_mask_outs_and_targets = imgs[:, 0, :, :] != DEFAULT_VAL
    no_pad_mask_outs_and_targets = no_pad_mask_outs_and_targets[:, None, :, :]
    return no_pad_mask_outs_and_targets


def apply_mask_normalization_2_input_get_mask_outputs(
    imgs: torch.tensor, selected_task: Task
):
    """Applies mask and normalization to the images:
      - puts -9999 pixels (added due to padding) to nans
      - applies normalization by ignoring nan values
      - set nan values to 0
    And returns the mask of not padded pixels that will be used
    to filter out padded pixels when computing the loss for semantic
    segmentation and concentration regression (not for image
    classification).

    Args:
        imgs (torch.tensor): batch of images N,C,H,W
        selected_task (Task): classification, sem_segm, or
          concentration regr.

    Returns:
        _type_: _description_
    """
    # Get mask of padding pixels
    padding_mask_inputs = get_padding_mask_inputs(imgs)
    if (
        selected_task == Task.SEM_SEGMENTATION
        or selected_task == Task.CONCENTRATION_REGRESSION
    ):
        # Get mask for ingoring padding in outputs and targets (only for sem segm and regr)
        no_pad_mask_outs_and_targets = get_no_padding_mask_outs_targets(imgs)

    # Normalize images
    imgs = standardise_imgs(imgs)
    # NOPE normalize in [0.1, 1]
    # imgs = 0.1 + 0.9 * (imgs - imgs.min()) / (imgs.max() - imgs.min())
    # Set default values to 0 for gradient purposes
    imgs[padding_mask_inputs] = 0
    if (
        selected_task == Task.SEM_SEGMENTATION
        or selected_task == Task.CONCENTRATION_REGRESSION
    ):
        return imgs, no_pad_mask_outs_and_targets
    else:
        return imgs


def apply_mask_2_input_get_mask_outputs(
    imgs: torch.tensor, selected_task: Task
):
    """Applies mask and normalization to the images:
      - puts -9999 pixels (added due to padding) to nans
      - applies normalization by ignoring nan values
      - set nan values to 0
    And returns the mask of not padded pixels that will be used
    to filter out padded pixels when computing the loss for semantic
    segmentation and concentration regression (not for image
    classification).

    Args:
        imgs (torch.tensor): batch of images N,C,H,W
        selected_task (Task): classification, sem_segm, or
          concentration regr.

    Returns:
        _type_: _description_
    """
    # Get mask of padding pixels
    padding_mask_inputs = get_padding_mask_inputs(imgs)
    if (
        selected_task == Task.SEM_SEGMENTATION
        or selected_task == Task.CONCENTRATION_REGRESSION
    ):
        # Get mask for ingoring padding in outputs and targets (only for sem segm and regr)
        no_pad_mask_outs_and_targets = get_no_padding_mask_outs_targets(imgs)

    # Set default values to 0 for gradient purposes
    imgs[padding_mask_inputs] = 0
    if (
        selected_task == Task.SEM_SEGMENTATION
        or selected_task == Task.CONCENTRATION_REGRESSION
    ):
        return imgs, no_pad_mask_outs_and_targets
    else:
        return imgs


def standardise_imgs(image_tiles, norm_stat_dict=data_stats):
    """
    Standardise a batch of image tiles during training

    Parameters:
    - image_tiles(torch.Tensor): a batch of image tiles with shape (n_image, n_bands, n_pixel, n_pixel)
    - norm_stat_dict(dict): a dictionary containing pre-computed normalisation values for image and plume tiles

    Returns:
    - (torch.Tensor): a batch of standardised image tiles where each pixel is (value - img_mean) / img_std
    """
    num_bands = 86
    norm_imgs = []
    for b in range(num_bands):
        img_mean = norm_stat_dict["mean_tiles"][b]
        img_std = norm_stat_dict["std_tiles"][b]
        norm_imgs.append((image_tiles[:, b, :, :] - img_mean) / img_std)
    return torch.stack(norm_imgs, dim=1)


def standardise_plumes(plume_tiles, norm_stat_dict=data_stats):
    """
    Standardise a batch of plume tiles during training

    Parameters:
    - plume_tiles(torch.Tensor): a batch of plume tiles with shape (n_plume, 1, n_pixel, n_pixel)
    - norm_stat_dict(dict): a dictionary containing pre-computed normalisation values for image and plume tiles

    Returns:
    - (torch.Tensor): a batch of standardised plume tiles where each pixel is (value - plume_mean) / plume_std
    """
    plume_mean = norm_stat_dict["mean_plumes"]
    plume_std = norm_stat_dict["std_plumes"]
    plume_tiles = (plume_tiles - plume_mean) / plume_std

    return plume_tiles


def save_scene_ids_distribution_subsets(
    train_tile_paths, val_tile_paths, test_tile_paths, timestamp_training
) -> None:

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)

    lat_lon_pattern = re.compile(r"lat_lon_([-+]?\d+)P(\d+)_([-+]?\d+)P(\d+)")

    coords = defaultdict(set)
    colors = ["blue", "red", "green"]
    plume_flag = "p1"

    plume_train = 0
    no_plume_train = 0
    plume_val = 0
    no_plume_val = 0
    plume_test = 0
    no_plume_test = 0

    for filename in train_tile_paths:
        if plume_flag in filename:
            plume_train += 1
        else:
            no_plume_train += 1
        coord_match = lat_lon_pattern.search(filename)
        lat_int, lat_dec, lon_int, lon_dec = coord_match.groups()
        lat = float(f"{lat_int}.{lat_dec}")
        lon = float(f"{lon_int}.{lon_dec}")
        coords["train"].add((lat, lon))

    for filename in val_tile_paths:
        if plume_flag in filename:
            plume_val += 1
        else:
            no_plume_val += 1
        coord_match = lat_lon_pattern.search(filename)
        lat_int, lat_dec, lon_int, lon_dec = coord_match.groups()
        lat = float(f"{lat_int}.{lat_dec}")
        lon = float(f"{lon_int}.{lon_dec}")
        coords["val"].add((lat, lon))

    for filename in test_tile_paths:
        if plume_flag in filename:
            plume_test += 1
        else:
            no_plume_test += 1
        coord_match = lat_lon_pattern.search(filename)
        lat_int, lat_dec, lon_int, lon_dec = coord_match.groups()
        lat = float(f"{lat_int}.{lat_dec}")
        lon = float(f"{lon_int}.{lon_dec}")
        coords["test"].add((lat, lon))

    logger.info(f"Plumes in train: {plume_train}")
    logger.info(f"No plumes in train: {no_plume_train}")
    logger.info(f"Plumes in val: {plume_val}")
    logger.info(f"No plumes in val: {no_plume_val}")
    logger.info(f"Plumes in test: {plume_test}")
    logger.info(f"No plumes in test: {no_plume_test}")

    # lats, lons = [list(group) for group in zip(*coords)]
    types_set = ["train", "val", "test"]
    for color, set_type in zip(colors, types_set):
        coord = list(coords[set_type])
        lats, lons = zip(*coord)
        ax.scatter(
            lons,
            lats,
            color=color,
            s=30,
            alpha=0.4,
            label=str(set_type),
            transform=ccrs.PlateCarree(),
        )

    plt.legend()
    fig.savefig(
        os.path.join(
            ROOT_DIR,
            f"results/plots/scene_ids_distribution_subsets_{timestamp_training}.png",
        )
    )
    fig.clf()


def concentration_weighted_bce(
    pred, target, concentration, max_concentration=19383682.0, alpha=1.0
):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    methane_penalty = 1 + alpha * (concentration / max_concentration)
    return (bce_loss * methane_penalty).mean()


def inv_concentration_weighted_bce(
    pred, target, concentration, max_concentration=19383682.0, alpha=1.0
):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    methane_penalty = 1 - alpha * (concentration / max_concentration)
    return (bce_loss * methane_penalty).mean()
