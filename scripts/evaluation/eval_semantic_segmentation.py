import os
import sys
import torch
from torchmetrics.segmentation import MeanIoU
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import Accuracy
import numpy as np
from tqdm import tqdm
from tomlkit import load
from torch.utils.data import DataLoader
from loguru import logger
import matplotlib.pyplot as plt

sys.path.append(".")

from plume_hunter.dataset import PlumeHunterDataset, Task
from plume_hunter.models.sem_segmentation.unet import UNet
from plume_hunter.constants import ROOT_DIR
from scripts.training.train_utils import (
    apply_mask_normalization_2_input_get_mask_outputs,
    save_scene_ids_distribution_subsets,
)
from scripts.training.parse_args_train import parse_args_train
from scripts.training.set_seed import set_seed


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

    use_cuda = not config["training"]["no_cuda"] and torch.cuda.is_available()

    set_seed(config["reproducibility"]["seed"])
    data_dir = config["data"]["data_dir"]

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device in use: {device}")

    train_kwargs = {}
    test_kwargs = {}
    num_workers = config["training"]["num_workers"]

    if use_cuda:
        cuda_kwargs = {
            "num_workers": num_workers,
            "pin_memory": False,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    num_bands = config["data"]["num_bands"]
    num_classes = config["data"]["num_classes"]

    # Metrics sem segm
    mean_iou = MeanIoU(num_classes=num_classes - 1)
    precision = Precision(num_classes=num_classes - 1, task="binary")
    recall = Recall(num_classes=num_classes - 1, task="binary")
    f1_score = F1Score(num_classes=num_classes - 1, task="binary")

    mean_iou.to(device)
    precision.to(device)
    recall.to(device)
    f1_score.to(device)
    # Metrics img class
    accuracy = Accuracy(task="binary", num_classes=num_classes - 1)
    img_c_prec = Precision(task="binary", num_classes=num_classes - 1)
    img_c_rec = Recall(task="binary", num_classes=num_classes - 1)
    img_c_f1 = F1Score(task="binary", num_classes=num_classes - 1)

    accuracy.to(device)
    img_c_prec.to(device)
    img_c_rec.to(device)
    img_c_f1.to(device)

    test_scene_ids = np.load(
        os.path.join(ROOT_DIR, "scripts", "training", "scene_ids_test.npy")
    ).tolist()

    test_dataset = PlumeHunterDataset(
        data_dir=data_dir,
        orthorectified=config["data"]["orthorectified"],
        scene_ids=test_scene_ids,
        task=selected_task,
        subset=False,
        extra_jitt=False,
        augm_data=False,
        only_plumes=False,
        exclude_corrupted=config["data"]["exclude_corrupted"],
        balanced=True,
    )

    logger.info(f"{len(test_dataset)} tiles in the test dataset")

    if plot_distrib:
        save_scene_ids_distribution_subsets(
            train_tile_paths=test_dataset.tile_files,
            val_tile_paths=test_dataset.tile_files,
            test_tile_paths=test_dataset.tile_files,
            timestamp_training="IGNORE_PLOT_eval_whole_test",
        )

    # Keep batch_size = 1 to make the image classification metrics
    # computable.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
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

    model_id = config["eval"]["model_id"]

    model_dir = config["save"]["save_dir"]
    model_path = os.path.join(model_dir, model_id)

    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Evaluate model on normalised test set
    # And also perform image classification based on the num of pixels
    # predicted as plume.
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs, targets, concentr, name_tile = batch
            imgs, targets, concentr = (
                imgs.to(device),
                targets.to(device),
                concentr.to(device),
            )
            imgs, no_pad_mask_outs_and_targets = (
                apply_mask_normalization_2_input_get_mask_outputs(
                    imgs, selected_task
                )
            )

            outputs = model(imgs)
            outputs_masked = outputs[no_pad_mask_outs_and_targets]

            MIN_CONCENTR = -900000  # 900

            target_masked = targets[no_pad_mask_outs_and_targets]
            # Whole plume thresh:
            # Consider only plumes whose max concentr > thresh
            # and whose num of plume pixels > another thresh
            if concentr.max() < MIN_CONCENTR and target_masked.sum() > 0:
                continue

            # Metrics for sem segm
            one_hot_pred_masked = (torch.sigmoid(outputs_masked) > 0.5).int()
            one_hot_target_masked = target_masked.int()
            mean_iou.update(one_hot_pred_masked, one_hot_target_masked)
            precision.update(one_hot_pred_masked, one_hot_target_masked)
            recall.update(one_hot_pred_masked, one_hot_target_masked)
            f1_score.update(one_hot_pred_masked, one_hot_target_masked)

            # _____Metrics for image classification_____
            # At least 1 pixel predicted as plume
            # => pred 4 img class = 1 ,
            # 0 otherwise
            if one_hot_pred_masked.sum() > 0:
                img_class_pred = 1
            else:
                img_class_pred = 0
            img_class_pred = torch.tensor([img_class_pred])
            # At least 1 pixel of ground truth as plume
            # => target 4 img class = 1,
            # 0 otherwise
            if one_hot_target_masked.sum() > 0:
                img_class_target = 1
            else:
                img_class_target = 0
            img_class_target = torch.tensor([img_class_target])

            preds.append(img_class_pred)
            truths.append(img_class_target)

            accuracy.update(img_class_pred, img_class_target)
            img_c_prec.update(img_class_pred, img_class_target)
            img_c_rec.update(img_class_pred, img_class_target)
            img_c_f1.update(img_class_pred, img_class_target)

    # Semantic segm
    t_miou = mean_iou.compute()
    t_prec = precision.compute()
    t_rec = recall.compute()
    t_f1 = f1_score.compute()

    logger.info(
        (
            "Semantic segmentation - mIoU, precision, recall, f1-score on "
            f"Test Set: {t_miou * 100}, {t_prec * 100}, {t_rec * 100}, "
            f"{t_f1 * 100}"
        )
    )
    # Image classification
    truths = np.array(truths).flatten()
    preds = np.array(preds).flatten()

    acc_class_res = accuracy.compute()
    prec_class_res = img_c_prec.compute()
    rec_class_res = img_c_rec.compute()
    f1_class_res = img_c_f1.compute()

    logger.info(
        (
            "Image classification - accuracy, precision, recall, f1-score on Test "
            f"Set: {acc_class_res * 100}, {prec_class_res * 100}, "
            f"{rec_class_res * 100}, {f1_class_res * 100}"
        )
    )

    no_plume_preds = [preds[truths == 0]]
    plume_preds = [preds[truths == 1]]

    # Plot outputs depending on the ML task
    fig, ax = plt.subplots(figsize=(11, 9))
    binning = np.linspace(0, 1, 50)
    ax.hist(
        no_plume_preds,
        bins=binning,
        label="No plumes",
        density=True,
        alpha=0.4,
    )
    ax.hist(
        plume_preds, bins=binning, label="With plumes", density=True, alpha=0.4
    )
    ax.set_xlabel("Classification outputs", loc="right")
    ax.set_ylabel("Normalised frequency", loc="top")
    ax.legend(loc="upper right")
    fig.savefig("classification_outputs.pdf")


if __name__ == "__main__":
    options = parse_args_train()
    main(options)
