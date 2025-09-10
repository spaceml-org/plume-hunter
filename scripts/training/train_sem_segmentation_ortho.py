import os

import numpy as np
import torch
import torch.optim as optim
import wandb
from tomlkit import load
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
from torchmetrics import Precision, Recall, F1Score
from tqdm import tqdm
import kornia.augmentation as K
from loguru import logger
import sys

from plume_hunter.constants import ROOT_DIR
from plume_hunter.dataset import PlumeHunterDataset, Task
from plume_hunter.models.sem_segmentation.unet import UNet
from plume_hunter.loss.dice_loss import DiceLoss

from set_seed import set_seed
from parse_args_train import parse_args_train
from train_utils import (
    log_sem_segmentation,
    get_model_name_task_timestamp,
    apply_mask_normalization_2_input_get_mask_outputs,
    save_scene_ids_distribution_subsets,
)
from train_val_test_split import (
    train_val_test_location,
)

##############
# LOGGING
LOGLEVEL = "INFO"
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    level=LOGLEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green>|<blue>{level}</blue>|"
        " <level>{message}</level>"
    ),
)
logger.warning(f"Setting LogLevel to {LOGLEVEL}")


def main(options):
    plot_distrib = True

    config_path = options["config_path"]
    with open(config_path, "rb") as toml_file:
        config = load(toml_file)

    selected_task = Task.SEM_SEGMENTATION

    timestamp_training = get_model_name_task_timestamp(selected_task)

    if config["data"]["orthorectified"]:
        type_data = "Ortho"
    else:
        type_data = "Unortho"

    if config["log"]["wandb"]:
        wandb.init(
            project="yourprojectname",
            entity="yourentityname",
            config=config,
            name=f"{type_data}_{timestamp_training}",
        )
        logger.info("WandB initialized.")

    use_cuda = not config["training"]["no_cuda"] and torch.cuda.is_available()

    set_seed(config["reproducibility"]["seed"])

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Device in use: {device}")

    train_kwargs = {}
    # TODO: change shuffle to False
    test_kwargs = {"shuffle": True}
    sep = "_" * 20
    num_workers = config["training"]["num_workers"]

    if use_cuda:
        cuda_kwargs = {
            "num_workers": num_workers,
            "pin_memory": False,
        }  # , "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    saved_models_folder_path = os.path.join(
        ROOT_DIR,
        "results",
        "trained_models",
    )
    if not os.path.exists(saved_models_folder_path):
        os.makedirs(saved_models_folder_path)

    plots_folder_path = os.path.join(
        ROOT_DIR,
        "results",
        "plots",
    )
    if not os.path.exists(plots_folder_path):
        os.makedirs(plots_folder_path)

    batch_size = config["training"]["batch_size"]
    test_batch_size = config["training"]["test_batch_size"]
    data_dir = config["data"]["data_dir"]

    num_bands = config["data"]["num_bands"]
    num_classes = config["data"]["num_classes"]
    # Initialize metrics based on selected task

    mean_iou = MeanIoU(num_classes=num_classes - 1)
    precision = Precision(num_classes=num_classes - 1, task="binary")
    recall = Recall(num_classes=num_classes - 1, task="binary")
    f1_score = F1Score(num_classes=num_classes - 1, task="binary")

    mean_iou.to(device)
    precision.to(device)
    recall.to(device)
    f1_score.to(device)

    # Create train/val/test split
    path_tiles = os.path.join(
        config["data"]["data_dir"], "tiles_orthorectified/tiles"
    )

    train_scene_ids, val_scene_ids, test_scene_ids = train_val_test_location(
        path_tiles=path_tiles,
        cell_km_geogr_split=config["data"]["cell_km_geogr_split"],
        seed=config["reproducibility"]["seed"],
        train_size=config["data"]["train_size"],
        perc_val_compared_to_val_plus_test=config["data"][
            "perc_val_compared_to_val_plus_test"
        ],
    )

    # Random split
    # Creating/reading pkl file containing image->label tile mapping
    train_val_dataset = PlumeHunterDataset(
        data_dir=data_dir,
        scene_ids=train_scene_ids + val_scene_ids,  # train_scene_ids,
        task=selected_task,
        subset=False,
        extra_jitt=config["data"]["extra_jitt"],
        augm_data=config["data"]["augm_data"],
        only_plumes=config["data"]["only_plumes"],
        exclude_corrupted=config["data"]["exclude_corrupted"],
        orthorectified=config["data"]["orthorectified"],
    )
    print(test_scene_ids)
    test_dataset = PlumeHunterDataset(
        data_dir=data_dir,
        scene_ids=test_scene_ids,
        task=selected_task,
        subset=False,
        extra_jitt=False,
        augm_data=False,
        only_plumes=config["data"]["only_plumes"],
        exclude_corrupted=config["data"]["exclude_corrupted"],
        orthorectified=config["data"]["orthorectified"],
    )

    # 0.889 of train_val * 0.9 (train_val) = 0.8 -> training contains 80%
    # of the total tiles
    perc_train_wrt_train_val = (config["data"]["train_size"] * 100) / (
        100 - ((1 - config["data"]["train_size"]) / 2) * 100
    )

    train_size = int(perc_train_wrt_train_val * len(train_val_dataset))
    valid_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, valid_size]
    )

    transform = torch.nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation90(times=(1, 3), p=0.5),
    )

    train_tiles = np.array(train_val_dataset.tile_files)[
        train_dataset.indices
    ].tolist()
    val_tiles = np.array(train_val_dataset.tile_files)[
        val_dataset.indices
    ].tolist()

    """
    train_plus_val_scene_ids = train_val_dataset.get_scene_ids(
        np.array(train_val_dataset.tile_files)
    )
    final_train_scene_ids = train_val_dataset.get_scene_ids(
        np.array(train_val_dataset.tile_files)[train_dataset.indices]
    )
    final_val_scene_ids = train_val_dataset.get_scene_ids(
        np.array(train_val_dataset.tile_files)[val_dataset.indices]
    )

    np.save("tiles_train.npy", train_tiles)
    np.save("tiles_val.npy", val_tiles)
    np.save("tiles_test.npy", test_dataset.tile_files)

    np.save("scene_ids_train.npy", final_train_scene_ids)
    np.save("scene_ids_val.npy", final_val_scene_ids)
    np.save("scene_ids_test.npy", test_scene_ids)
    """

    if plot_distrib:
        save_scene_ids_distribution_subsets(
            train_tile_paths=train_tiles,
            val_tile_paths=val_tiles,
            test_tile_paths=test_dataset.tile_files,
            timestamp_training=timestamp_training,
        )

    logger.info(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, "
        f"Test size: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **train_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        drop_last=False,
        **test_kwargs,
    )

    model_arch = config["training"]["model_arch"]
    if model_arch == "UNet":
        model = UNet(
            input_bands=num_bands,
            output_classes=num_classes - 1,
            hidden_channels=config["training"]["hidden_channels"],
        ).to(device)
    else:
        logger.warning("Please specify a valid model architecture (CNN, ViT)")
    logger.info(f"Model: {model}")
    # load to file in case path is provided in config:
    if config["training"]["load_model_path"]:
        model_path = config["training"]["load_model_path"]
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.warning(
                f"Model path {model_path} does not exist. Starting training from scratch."
            )

    if config["training"]["optimizer_name"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            # weight_decay=config["training"]["weight_decay"],
        )
    elif config["training"]["optimizer_name"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

    criterion = DiceLoss()

    logger.info(f"Loss: {str(criterion)}")

    if config["training"]["scheduler"]:
        # CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=15, eta_min=0.001
        )

    best_val_loss = float("inf")

    best_model_path = os.path.join(
        saved_models_folder_path,
        f"{type_data}_{config["training"]["model_arch"]}_{timestamp_training}.pth",
    )

    curr_lr = config["training"]["lr"]
    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"{sep}Training - Epoch {epoch}: {sep}")
        model.train()
        running_loss = 0
        total_imgs = 0
        for imgs, targets, _, _ in tqdm(train_loader):

            optimizer.zero_grad()

            imgs, targets = imgs.to(device), targets.to(device)

            # Apply transform
            # stacked = torch.cat([imgs, targets], dim=1)  # (B, 87, H, W)
            # augmented = transform(stacked)

            # Split back
            # imgs = augmented[:, :num_bands, :, :]
            # targets = augmented[:, num_bands:, :, :]

            imgs, no_pad_mask_outs_and_targets = (
                apply_mask_normalization_2_input_get_mask_outputs(
                    imgs, selected_task
                )
            )

            outputs = model(imgs)

            outputs_masked = outputs[no_pad_mask_outs_and_targets]
            target_masked = targets[no_pad_mask_outs_and_targets]

            loss = criterion(
                outputs_masked.float(),
                target_masked.float(),
            )
            running_loss += loss.item() * imgs.shape[0]
            loss.backward()
            optimizer.step()
            # Calculate metrics
            total_imgs += imgs.shape[0]

            one_hot_pred_masked = (torch.sigmoid(outputs_masked) > 0.5).int()
            one_hot_target_masked = target_masked.int()
            mean_iou.update(
                one_hot_pred_masked,
                one_hot_target_masked,
            )
            precision.update(
                one_hot_pred_masked,
                one_hot_target_masked,
            )
            recall.update(
                one_hot_pred_masked,
                one_hot_target_masked,
            )
            f1_score.update(
                one_hot_pred_masked,
                one_hot_target_masked,
            )
            if config["log"]["wandb"]:
                wandb.log(
                    {
                        "train_loss_batch": loss.item(),
                    }
                )

        if config["training"]["scheduler"]:
            scheduler.step()  # scheduler update
            curr_lr = scheduler.optimizer.param_groups[0]["lr"]

        t_loss = running_loss / total_imgs

        t_miou = mean_iou.compute()
        t_prec = precision.compute()
        t_rec = recall.compute()
        t_f1 = f1_score.compute()

        log_sem_segmentation(
            train=True,
            logger=logger,
            wandb=wandb,
            epoch=epoch,
            curr_lr=curr_lr,
            loss=t_loss,
            miou=t_miou,
            prec=t_prec,
            rec=t_rec,
            f1=t_f1,
        )
        mean_iou.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()

        # ______Evaluation on Validation Set______
        model.eval()
        with torch.no_grad():
            running_loss = 0
            total_imgs = 0
            for v_imgs, v_targets, _, _ in val_loader:
                v_imgs, v_targets = v_imgs.to(device), v_targets.to(device)

                v_imgs, no_pad_mask_outs_and_targets = (
                    apply_mask_normalization_2_input_get_mask_outputs(
                        v_imgs, selected_task
                    )
                )

                v_outputs = model(v_imgs)

                v_outputs_masked = v_outputs[no_pad_mask_outs_and_targets]
                v_target_masked = v_targets[no_pad_mask_outs_and_targets]

                v_loss = criterion(
                    v_outputs_masked.float(),
                    v_target_masked.float(),
                )
                running_loss += v_loss.item() * v_imgs.shape[0]

                total_imgs += imgs.shape[0]

                v_one_hot_pred_masked = (
                    torch.sigmoid(v_outputs_masked) > 0.5
                ).int()
                v_one_hot_target_masked = v_target_masked.int()

                mean_iou.update(
                    v_one_hot_pred_masked,
                    v_one_hot_target_masked,
                )
                precision.update(
                    v_one_hot_pred_masked, v_one_hot_target_masked
                )
                recall.update(v_one_hot_pred_masked, v_one_hot_target_masked)
                f1_score.update(v_one_hot_pred_masked, v_one_hot_target_masked)

                if config["log"]["wandb"]:
                    wandb.log(
                        {
                            "val_loss_batch": v_loss.item(),
                        }
                    )

            v_loss = running_loss / total_imgs

            v_miou = mean_iou.compute()
            v_prec = precision.compute()
            v_rec = recall.compute()
            v_f1 = f1_score.compute()

            log_sem_segmentation(
                train=False,
                logger=logger,
                wandb=wandb,
                epoch=epoch,
                curr_lr=curr_lr,
                loss=v_loss,
                miou=v_miou,
                prec=v_prec,
                rec=v_rec,
                f1=v_f1,
            )

            mean_iou.reset()
            precision.reset()
            recall.reset()
            f1_score.reset()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                save_path = (
                    best_model_path + "_" + "epoch" + "_" + str(epoch) + ".pth"
                )
                torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    options = parse_args_train()
    main(options)
