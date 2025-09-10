import os
import argparse

from plume_hunter.constants import ROOT_DIR


def parse_args_train():
    parser = argparse.ArgumentParser()
    # Options
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path of the config file containing parameters and their values",
    )

    # Disable during debugging @ Run through terminal
    args = parser.parse_args()
    # """Disable when run through terminal: For debugging process
    # """

    # convert to ordinary dict
    options = vars(args)

    return options
