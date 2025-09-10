import os
import glob
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import math

TILES_EXT = "*.npy"


def train_val_test_location(
    path_tiles: str,
    cell_km_geogr_split: int,
    train_size: float,
    perc_val_compared_to_val_plus_test: float,
    seed: int,
):
    CELL_SIZE_KM = cell_km_geogr_split
    DEG_LAT = 0.009 * CELL_SIZE_KM
    DEG_LON = 0.009 * CELL_SIZE_KM
    scene_ids = []
    lats = []
    lons = []
    train_scene_ids = []
    val_scene_ids = []
    test_scene_ids = []

    tiles_path = glob.glob(os.path.join(path_tiles, TILES_EXT))

    for path_tile in tiles_path:
        tokens = path_tile.split("/")[-1].split("_")
        scene_id = "_".join(tokens[:3])
        lat = float(tokens[-6].replace("P", "."))
        lon = float(tokens[-5].replace("P", "."))

        scene_ids.append(scene_id)
        lats.append(lat)
        lons.append(lon)

    # Get unique scene ids
    unique_scene_ids = list(set(scene_ids))
    unique_scene_ids, idxs_unique_scene_ids = np.unique(
        scene_ids, return_index=True
    )

    lats_unique = np.array(lats)[idxs_unique_scene_ids]
    lons_unique = np.array(lons)[idxs_unique_scene_ids]

    grid_cells = defaultdict(list)

    for i in range(len(unique_scene_ids)):
        lat, lon = lats_unique[i], lons_unique[i]
        lat_key = int(lat / DEG_LAT)
        lon_key = int(lon / (DEG_LON / math.cos(math.radians(lat))))
        grid_cells[(lat_key, lon_key)].append(unique_scene_ids[i])

    # Determine how many steps in each cycle
    cycle_length = round(
        1
        / min(
            perc_val_compared_to_val_plus_test,
            1 - perc_val_compared_to_val_plus_test,
        )
    )
    count_val = round(perc_val_compared_to_val_plus_test * cycle_length)
    count_test = cycle_length - count_val

    # Build the pattern: True for val, False for test
    pattern = [True] * int(count_val) + [False] * int(count_test)
    i = 0

    for _, images in grid_cells.items():
        if len(images) == 1:
            # Single image goes to training
            train_scene_ids.extend(images)
        else:
            # Multiple images: split into train and temp
            train, temp = train_test_split(
                images, train_size=train_size, random_state=seed
            )
            train_scene_ids.extend(train)

            # Now handle the remaining images (temp)
            if len(temp) == 1:
                # Single remaining image: use pattern to decide val vs test
                if pattern[i % len(pattern)]:
                    val_scene_ids.extend(temp)
                else:
                    test_scene_ids.extend(temp)
                i += 1
            elif len(temp) > 1:
                # Multiple remaining images: split between val and test
                val, test = train_test_split(
                    temp,
                    train_size=perc_val_compared_to_val_plus_test,
                    random_state=seed,
                )
                val_scene_ids.extend(val)
                test_scene_ids.extend(test)

    return train_scene_ids, val_scene_ids, test_scene_ids


def train_val_test_based_on_time(
    path_tiles: str,
    latest_year_train_incl: int,
    latest_month_train_incl: int,
    perc_val_compared_to_val_plus_test: float = 0.5,
) -> tuple[list[str], list[str], list[str]]:
    """Gets the list of scene ids from tiles, takes the unique ones,
    and split them into train, val and test splits them temporally.
    Put all tiles before and including the specified year and month in train,
    and divide the rest based on the specified percentage in val and test.

    Args:
        path_tiles (str): path of folder containing tiles.
        latest_year_train_incl (int): specified year (included).
        latest_month_train_incl (int): specified month (included).
        perc_val_compared_to_val_plus_test (float, optional): specified
          percentage of val tiles to get from the tiles that remain after
          getting the train tiles. If 0.5, then half of remaining tiles
          will be in val set and the other half in test set.
          Defaults to 0.5.

    Returns:
        tuple[list[str], list[str], list[str]]: scene ids in train, val, and
          test sets.
    """

    scene_ids = []
    train_scene_ids = []
    val_scene_ids = []
    test_scene_ids = []
    val_test_ids = []

    tiles_path = glob.glob(os.path.join(path_tiles, TILES_EXT))

    for path_tile in tiles_path:
        scene_id = "_".join(path_tile.split("/")[-1].split("_")[:3])
        scene_ids.append(scene_id)

    # Get unique scene ids
    unique_scene_ids = list(set(scene_ids))

    for scene_id in unique_scene_ids:
        yyyymmdd = scene_id.split("_")[0].split("T")[0]
        year = int(yyyymmdd[:4])
        month = int(yyyymmdd[4:6])
        if year < latest_year_train_incl or (
            year == latest_year_train_incl and month <= latest_month_train_incl
        ):
            train_scene_ids.append(scene_id)
        else:
            val_test_ids.append(scene_id)

    # Sort remaining scene_ids (based on time), and divide latest months of
    # data into val and test sets
    val_test_ids.sort()

    # Calculate split index
    split_index = int(len(val_test_ids) * perc_val_compared_to_val_plus_test)

    # Divide the list
    val_scene_ids = val_test_ids[:split_index]
    test_scene_ids = val_test_ids[split_index:]

    return train_scene_ids, val_scene_ids, test_scene_ids


def train_val_test_random_split_scene_ids(
    path_tiles: str, train_perc: float, val_perc: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Gets the list of scene ids from tiles, takes the unique ones,
    and split them randomly into train, val and test splits based on the
    indicated percentages.

    Args:
        path_tiles (str): path of folder containing tiles.
        train_perc (float): percentage of scene ids to be contained in train
          set.
        val_perc (float): percentage of scene ids to be contained in val
          set.
        seed (int): seed.

    Returns:
        tuple[list[str], list[str], list[str]]: scene ids in train, val, and
          test sets.
    """

    test_perc = 1.0 - (train_perc + val_perc)
    assert train_perc + val_perc + test_perc == 1.0
    scene_ids = []
    tiles_path = glob.glob(os.path.join(path_tiles, TILES_EXT))

    for path_tile in tiles_path:
        scene_id = "_".join(path_tile.split("/")[-1].split("_")[:3])
        scene_ids.append(scene_id)

    # Get unique scene ids
    unique_scene_ids = list(set(scene_ids))

    # Step 1: Split into N% train and 1-N% temp
    train_scene_ids, temp_files = train_test_split(
        unique_scene_ids, test_size=(val_perc + test_perc), random_state=seed
    )

    test_ratio_within_temp = test_perc / (val_perc + test_perc)

    # Step 2: Split temp into 1-M% val and M% test → gives 10% each of total
    val_scene_ids, test_scene_ids = train_test_split(
        temp_files, test_size=test_ratio_within_temp, random_state=seed
    )

    return train_scene_ids, val_scene_ids, test_scene_ids
