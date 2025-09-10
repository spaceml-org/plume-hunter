from enum import Enum
import os
import random
import torch
import glob
import pickle
import numpy as np
from loguru import logger
from functools import lru_cache
from tqdm import tqdm
from torch.utils.data import Dataset

from plume_hunter.constants import ROOT_DIR


data_stats = {
    "mean_tiles": np.array(
        [
            8.820154,
            8.460156,
            8.378163,
            1.8905494,
            1.9696121,
            2.051052,
            2.0481863,
            1.8662518,
            1.8515557,
            2.0194266,
            2.0570755,
            2.0099096,
            1.898931,
            1.8895162,
            1.8999922,
            1.8546193,
            1.8227475,
            1.7899115,
            1.7599409,
            1.7175841,
            1.6734623,
            0.3026594,
            0.07638709,
            0.09836459,
            0.17778747,
            0.42972004,
            0.6042376,
            0.5990162,
            0.46444103,
            0.34571093,
            0.38242778,
            0.4192537,
            0.5531125,
            0.6049201,
            0.6351968,
            0.64347565,
            0.6605681,
            0.6478405,
            0.64813167,
            0.66844416,
            0.67105573,
            0.65474105,
            0.60297,
            0.58420557,
            0.5634987,
            0.571741,
            0.5483458,
            0.5458253,
            0.5061326,
            0.49481395,
            0.5313875,
            0.52744883,
            0.5215508,
            0.5078927,
            0.48895434,
            0.4678808,
            0.44340315,
            0.42657933,
            0.41545898,
            0.40868145,
            0.3927575,
            0.35643876,
            0.34498137,
            0.36877,
            0.32992765,
            0.30295795,
            0.32586485,
            0.28374532,
            0.24578913,
            0.22330448,
            0.26563954,
            0.22777179,
            0.19508593,
            0.18495494,
            0.18817358,
            0.20305926,
            0.18177232,
            0.14494999,
            0.123880036,
            0.16416314,
            0.13100469,
            0.13424091,
            0.07047546,
            0.09737421,
            0.08990364,
            0.058879077,
        ],
        dtype=np.float32,
    ),
    "std_tiles": np.array(
        [
            3.5184622,
            3.9359884,
            4.5602303,
            1.0194325,
            1.0539719,
            1.0885388,
            1.0814319,
            0.9844466,
            0.97067803,
            1.050371,
            1.0654551,
            1.0377113,
            0.97890604,
            0.9719128,
            0.9748114,
            0.95140296,
            0.9349948,
            0.9176357,
            0.90332747,
            0.8827087,
            0.8619241,
            0.1925423,
            0.051190775,
            0.065831386,
            0.11326674,
            0.26609284,
            0.37360862,
            0.3716854,
            0.29021788,
            0.21899162,
            0.24132077,
            0.2624092,
            0.34372404,
            0.37453198,
            0.3917388,
            0.39557827,
            0.4049414,
            0.3953669,
            0.3935973,
            0.4040568,
            0.40349588,
            0.39123914,
            0.357534,
            0.34391442,
            0.32957402,
            0.3324488,
            0.31671682,
            0.31306204,
            0.2890252,
            0.28044695,
            0.2990142,
            0.29782194,
            0.29589042,
            0.29000017,
            0.2813154,
            0.27176,
            0.2603568,
            0.2529959,
            0.24822015,
            0.24514389,
            0.23595876,
            0.2149928,
            0.20988537,
            0.22482668,
            0.2017001,
            0.18526435,
            0.19944172,
            0.17469995,
            0.15262763,
            0.13943072,
            0.1647267,
            0.14241616,
            0.123597704,
            0.120382085,
            0.12227614,
            0.13239524,
            0.11938096,
            0.10042191,
            0.08827085,
            0.111255184,
            0.09007388,
            0.09281237,
            0.0567022,
            0.072172746,
            0.06856111,
            0.050479643,
        ],
        dtype=np.float32,
    ),
    "min_tiles": np.array(
        [
            2.13498545e00,
            1.06006563e00,
            5.15686393e-01,
            1.79446067e-06,
            4.05045967e-06,
            1.34410020e-05,
            1.93630531e-05,
            6.59581701e-06,
            5.79211274e-08,
            1.76859212e-05,
            5.46465453e-05,
            4.18531727e-06,
            1.02809045e-05,
            6.90187971e-06,
            2.75053771e-05,
            4.38443240e-05,
            1.61741282e-05,
            4.28699313e-05,
            1.70001586e-05,
            9.42472980e-06,
            7.65566165e-06,
            9.64882041e-09,
            7.39457939e-10,
            3.93744121e-10,
            4.71954316e-11,
            4.10053191e-09,
            7.11693176e-08,
            5.32051914e-09,
            1.91971061e-09,
            3.03931769e-09,
            4.49663062e-09,
            1.63829714e-08,
            3.96077731e-08,
            2.94473956e-08,
            3.28128742e-08,
            3.77079914e-08,
            2.96549940e-09,
            2.81827770e-08,
            2.54887267e-09,
            1.08110861e-07,
            9.02487507e-09,
            2.24128645e-08,
            2.97772704e-08,
            1.15542500e-08,
            3.58706984e-08,
            1.75625150e-08,
            1.06453291e-08,
            2.08464641e-08,
            3.43336604e-09,
            2.20794538e-09,
            3.02033740e-08,
            1.39671661e-08,
            3.84353918e-08,
            8.13601275e-09,
            4.39870984e-09,
            1.31026283e-08,
            4.21872537e-09,
            8.38392999e-09,
            6.13843154e-09,
            5.51755308e-09,
            1.39492873e-09,
            7.16965110e-09,
            1.18058328e-08,
            1.20763444e-08,
            1.84726912e-09,
            9.07661768e-09,
            6.64104993e-10,
            5.24036947e-09,
            6.36083985e-09,
            2.45969856e-10,
            6.57889132e-10,
            2.06505568e-09,
            9.30513389e-10,
            4.27143876e-09,
            1.81456211e-10,
            2.14220108e-09,
            1.08531817e-09,
            3.51171936e-09,
            1.33133737e-09,
            7.20770776e-10,
            1.19040200e-09,
            7.46001261e-10,
            1.00651897e-10,
            6.86117829e-11,
            1.43386775e-10,
            9.29101018e-10,
        ],
        dtype=np.float32,
    ),
    "max_tiles": np.array(
        [
            1005.0469,
            1182.9216,
            1249.8956,
            259.16696,
            269.72208,
            281.05878,
            279.989,
            252.2973,
            249.90569,
            274.3049,
            279.25043,
            273.68307,
            257.52304,
            256.3786,
            258.47638,
            253.24109,
            247.6795,
            243.97969,
            239.935,
            237.15909,
            231.7069,
            39.037983,
            18.769424,
            10.29939,
            17.565023,
            57.952095,
            77.38035,
            74.50313,
            52.75174,
            33.782345,
            40.147896,
            46.562782,
            70.788185,
            77.87383,
            82.50287,
            84.03882,
            84.47553,
            83.275185,
            87.47167,
            95.53876,
            102.30135,
            99.42756,
            99.79621,
            97.45337,
            96.6521,
            94.84049,
            95.11181,
            93.998375,
            90.798195,
            88.25567,
            85.73581,
            81.42591,
            76.23715,
            68.82217,
            62.189533,
            56.48036,
            47.247948,
            44.537,
            42.8136,
            41.114403,
            38.605648,
            33.86047,
            31.121723,
            34.755314,
            29.265074,
            27.353985,
            31.29479,
            26.739433,
            23.970074,
            26.174747,
            30.227987,
            26.846817,
            31.11335,
            19.656197,
            19.734201,
            217.82106,
            18.917542,
            15.545634,
            13.490872,
            18.236221,
            15.007899,
            17.849953,
            22.21518,
            15.978986,
            15.292377,
            32.858376,
        ],
        dtype=np.float32,
    ),
    "q1_tiles": np.array(
        [
            6.5613155,
            5.5673995,
            4.759554,
            1.1422001,
            1.1989851,
            1.2582129,
            1.2628638,
            1.1524454,
            1.1500071,
            1.2628073,
            1.2911625,
            1.265338,
            1.1976702,
            1.1941069,
            1.2034793,
            1.1748422,
            1.1547433,
            1.1341743,
            1.1137187,
            1.0857892,
            1.0556052,
            0.15066414,
            0.03673589,
            0.04652995,
            0.08925444,
            0.2197124,
            0.3092973,
            0.30580246,
            0.23483884,
            0.17281663,
            0.19194683,
            0.21229227,
            0.28194007,
            0.30980036,
            0.32713124,
            0.33306715,
            0.34306973,
            0.33861294,
            0.34081727,
            0.3533461,
            0.35685655,
            0.35053858,
            0.3256836,
            0.31757072,
            0.3082179,
            0.31480768,
            0.30435285,
            0.30530852,
            0.28468326,
            0.28010583,
            0.3027174,
            0.30011773,
            0.29543057,
            0.28562784,
            0.27280092,
            0.2586221,
            0.2424042,
            0.23072723,
            0.22285351,
            0.21825545,
            0.20966709,
            0.19004458,
            0.18232895,
            0.19412722,
            0.17357226,
            0.16041309,
            0.17230228,
            0.14965983,
            0.1288284,
            0.11636539,
            0.1379809,
            0.11744087,
            0.099600285,
            0.092417195,
            0.093758866,
            0.10162372,
            0.08954532,
            0.06831305,
            0.056720696,
            0.077954315,
            0.061803572,
            0.062711164,
            0.029169422,
            0.04259878,
            0.038180057,
            0.022016939,
        ],
        dtype=np.float32,
    ),
    "q3_tiles": np.array(
        [
            10.406019,
            10.710393,
            11.31405,
            2.5745533,
            2.6770716,
            2.7810562,
            2.7727685,
            2.5255382,
            2.5017405,
            2.7231097,
            2.7707062,
            2.7048147,
            2.5542274,
            2.5399354,
            2.551466,
            2.4898973,
            2.446624,
            2.401709,
            2.3618965,
            2.303357,
            2.2444248,
            0.42795962,
            0.10723753,
            0.13991013,
            0.25159356,
            0.6118876,
            0.862243,
            0.85475403,
            0.66476333,
            0.49592316,
            0.5475845,
            0.59830934,
            0.78736633,
            0.85980296,
            0.9011061,
            0.9111988,
            0.935351,
            0.9154189,
            0.9144341,
            0.9425827,
            0.9451418,
            0.9208174,
            0.8450337,
            0.8182913,
            0.78851193,
            0.79859227,
            0.76353973,
            0.75880283,
            0.70221364,
            0.6858582,
            0.73573214,
            0.72972584,
            0.72201467,
            0.70410246,
            0.6785417,
            0.64996326,
            0.61689156,
            0.5944712,
            0.5798506,
            0.5709595,
            0.54842955,
            0.49699083,
            0.48216003,
            0.51612586,
            0.4609235,
            0.42140862,
            0.4539647,
            0.3944206,
            0.34096375,
            0.3093348,
            0.37102836,
            0.31762105,
            0.27165318,
            0.25714305,
            0.2625507,
            0.28416634,
            0.25411588,
            0.20201413,
            0.17293581,
            0.2308429,
            0.1835154,
            0.18857378,
            0.096946016,
            0.13638468,
            0.12577859,
            0.08125526,
        ],
        dtype=np.float32,
    ),
    "median_tiles": np.array(
        [
            8.3589115,
            8.020004,
            7.82612,
            1.7734382,
            1.8494774,
            1.9287996,
            1.9281005,
            1.7570267,
            1.7454574,
            1.9067382,
            1.9440762,
            1.9011679,
            1.7969955,
            1.7890579,
            1.8001206,
            1.7573098,
            1.7268263,
            1.6956155,
            1.6667591,
            1.6272272,
            1.5851804,
            0.28551954,
            0.06922088,
            0.088615626,
            0.16468519,
            0.4049146,
            0.56835467,
            0.56124485,
            0.43105644,
            0.3167716,
            0.3511283,
            0.386768,
            0.5126986,
            0.561087,
            0.58998525,
            0.5982499,
            0.6134777,
            0.6024682,
            0.60327584,
            0.62235165,
            0.6254654,
            0.6114026,
            0.56518126,
            0.5484741,
            0.5296482,
            0.53845537,
            0.51780903,
            0.51605004,
            0.47857115,
            0.4686801,
            0.5046019,
            0.49993256,
            0.49321622,
            0.4791111,
            0.46026576,
            0.43899256,
            0.414437,
            0.3977446,
            0.3870209,
            0.380446,
            0.36610562,
            0.33248538,
            0.32046613,
            0.34346354,
            0.30726594,
            0.28266713,
            0.30388948,
            0.26440147,
            0.22916967,
            0.20854856,
            0.24842553,
            0.21327768,
            0.18226556,
            0.1704185,
            0.1737199,
            0.18886541,
            0.16709648,
            0.129875,
            0.10936397,
            0.14876398,
            0.117915966,
            0.12054618,
            0.058246665,
            0.08440743,
            0.07670239,
            0.04700697,
        ],
        dtype=np.float32,
    ),
    "mean_plumes": 451.16068,
    "std_plumes": 527.42865,
    "min_plumes": 0.0,
    "max_plumes": 25326.93,
    "q1_plumes": 144.96578979492188,
    "q3_plumes": 578.223876953125,
    "median_plumes": 308.14908,
    "max_tile_concentration": 19383682.0,
}


class Task(Enum):
    CLASSIFICATION = 1
    SEM_SEGMENTATION = 2
    CONCENTRATION_REGRESSION = 3


class PlumeHunterDataset(Dataset):
    def __init__(
        self,
        scene_ids: list[str],
        data_dir,
        task=Task.CLASSIFICATION,
        orthorectified=True,
        input_output_mapping=None,
        subset=False,
        extra_jitt=True,
        augm_data=True,
        exclude_corrupted: bool = False,
        only_plumes: bool = False,
        balanced: bool = False,
    ):
        """
        Initialize the PlumeHunterDataset.
        Args:
            scene_ids (list[str]): list of scene ids to be considered in the
              dataset.
            data_dir (str): Path to the directory containing the dataset.
            orthorectified (bool): If True, use orthorectified tiles,
              otherwise use unorthorectified.
            task (Task): selected task between image classification, semantic
              segmentation, or regression of gas concentration.
            input_output_mapping (str): Path to the input/output mapping
              dictionary (if it does not exist it will be created on the fly).
            subset (bool): If True, use a subset of the data.
            extra_jitt (bool): If True, use also extra jittered data.
            augm_data (bool): If True, use also augmented data.
            only_plumes (bool): If True, keep only tiles with plumes.
            exclude_corrupted (bool): If True, excludes tiles not containing
              plumes from the  scene ids with nan values.
        """
        if orthorectified:
            if subset:
                data_dir = os.path.join(data_dir, "tiles_subsets")
            else:
                data_dir = os.path.join(data_dir, "tiles_orthorectified")
        else:
            if subset:
                data_dir = os.path.join(
                    data_dir, "tiles_subsets_unorthorectified"
                )
            else:
                data_dir = os.path.join(
                    data_dir,
                    "tiles_unorthorectified",
                )
        self.task = task
        if orthorectified:
            self.tiles_path = os.path.join(data_dir, "tiles")
            self.plumes_path = os.path.join(data_dir, "plumes")
        else:
            self.tiles_path = os.path.join(
                data_dir, "no_pad_no_jitter_no_corrupt", "tiles"
            )
            self.plumes_path = os.path.join(
                data_dir, "no_pad_no_jitter_no_corrupt", "plumes"
            )

        # Extra jitt
        self.extra_jitt = extra_jitt
        if self.extra_jitt:
            if orthorectified:
                folder_extra_jitt = "extra_jittering"
            else:
                folder_extra_jitt = "no_pad_with_jitter"
            self.extra_jitt_tiles_path = os.path.join(
                data_dir, folder_extra_jitt, "tiles"
            )
            self.extra_jitt_plumes_path = os.path.join(
                data_dir, folder_extra_jitt, "plumes"
            )
        # Augmented data
        self.augm_data = augm_data
        if self.augm_data:
            self.augm_data_tiles_path = os.path.join(
                data_dir, "augmented_data", "tiles"
            )
            self.augm_data_plumes_path = os.path.join(
                data_dir, "augmented_data", "plumes"
            )

        # try to read from the dictionary of input/output mapping
        self.input_output_mapping = input_output_mapping
        if self.input_output_mapping is not None and os.path.exists(
            self.input_output_mapping
        ):
            logger.info(
                "Trying to read from existing mapping file:",
                self.input_output_mapping,
            )
            with open(self.input_output_mapping, "rb") as f:
                self.mapping = pickle.load(f)
            logger.info("Mapping file found, using it.")
        else:
            if orthorectified:
                name_mapping_file = "input_tiles_output_plumes_mapping.pkl"
            else:
                name_mapping_file = (
                    "input_tiles_output_plumes_mapping_unorthorectified.pkl"
                )
            logger.info("Mapping file not found, creating a new mapping.")
            self.mapping = self._create_mapping(
                self.extra_jitt, self.augm_data
            )
            # save to pickle file:
            logger.info(
                "Saving the mapping to:",
                name_mapping_file,
            )
            with open(name_mapping_file, "wb") as f:
                pickle.dump(self.mapping, f)
        if scene_ids != None:
            # Keep only tiles in specified scene_ids
            filtered_paths = {
                key: value
                for key, value in self.mapping.items()
                if any(id_ in key for id_ in scene_ids)
            }
            self.mapping = filtered_paths
        if only_plumes:
            # Keep only tiles containing plumes
            filtered_paths = {
                key: value
                for key, value in self.mapping.items()
                if "p1" in key
            }
            self.mapping = filtered_paths
        if exclude_corrupted:
            corrupted_scenes = np.load(
                os.path.join(ROOT_DIR, "scripts", "training", "corrupt.npy"),
                mmap_mode="r",
            )
            corrupted_scenes = [
                "_".join(elem.split(".")[-2].split("_")[-3:])
                for elem in corrupted_scenes
            ]

            # Keep only not corrupted scene ids
            filtered_paths = {
                key: value
                for key, value in self.mapping.items()
                if not any(
                    scene_id in key and "p0" in key
                    for scene_id in corrupted_scenes
                )
            }
            self.mapping = filtered_paths
        # now let's extract the list of the tiles files:
        self.tile_files = list(self.mapping.keys())

        # If balanced, keeps 50% plumes and 50% no plumes
        if balanced:
            random.shuffle(self.tile_files)
            plumes_tmp = []
            count_p = 0
            for path in self.tile_files:
                if "p1" in path:
                    plumes_tmp.append(path)
                    count_p += 1
            count_no_p = 0
            no_plumes_tmp = []
            for path in self.tile_files:
                if "p0" in path and count_no_p < count_p:
                    no_plumes_tmp.append(path)
                    count_no_p += 1
                elif count_no_p == count_p:
                    break
            logger.info(f"Number of plumes: {count_p}")
            self.tile_files = plumes_tmp + no_plumes_tmp

        self.scene_ids = self.get_scene_ids(self.tile_files)

    def get_scene_ids(self, tile_files):
        scene_ids = []
        for file_tile in tile_files:
            scene_id = "_".join(file_tile.split("/")[-1].split("_")[:3])

            scene_ids.append(scene_id)
        unique_scene_ids = list(set(scene_ids))
        return unique_scene_ids

    def _extract_key(self, filename, augm=False):
        # Extracts a unique key from a filename: you can customize this
        parts = filename.split("_")
        if augm:
            id_part = "_".join(parts[1:4])
        else:
            id_part = "_".join(parts[:3])
        jittering_part = "_".join(parts[-4:])
        return f"{id_part}_{jittering_part}"

    def _create_mapping(self, extra_jitt, augm_data):
        tiles_path = self.tiles_path
        plumes_path = self.plumes_path
        # Step 1: Index all plume files by their matching key
        plume_files = os.listdir(plumes_path)
        plume_dict = {}

        for filename in plume_files:
            key = self._extract_key(filename)
            plume_dict[key] = os.path.join(plumes_path, filename)

        # Step 2: Match each tile file using the key
        tile_files = os.listdir(tiles_path)
        mapping = {}

        for filename in tqdm(tile_files, desc="Mapping tiles to plumes"):
            key = self._extract_key(filename)
            tile_path = os.path.join(tiles_path, filename)

            plume_path = plume_dict.get(key)
            if plume_path:
                mapping[tile_path] = plume_path
            else:
                logger.info(f"No plume found for tile {tile_path}")

        if extra_jitt:
            extra_jitt_plume_files = os.listdir(self.extra_jitt_plumes_path)
            for filename in extra_jitt_plume_files:
                key = self._extract_key(filename)
                plume_dict[key] = os.path.join(
                    self.extra_jitt_plumes_path, filename
                )

            extra_jitt_tile_files = os.listdir(self.extra_jitt_tiles_path)
            for filename in tqdm(
                extra_jitt_tile_files,
                desc="Mapping extra jittered tiles to plumes",
            ):
                key = self._extract_key(filename)
                extra_jitt_tile_path = os.path.join(
                    self.extra_jitt_tiles_path, filename
                )

                extra_jitt_plume_path = plume_dict.get(key)
                if extra_jitt_plume_path:
                    mapping[extra_jitt_tile_path] = extra_jitt_plume_path
                else:
                    logger.info(
                        f"No extra jittered plume found for tile {extra_jitt_tile_path}"
                    )

        if augm_data:
            augm_data_plume_files = os.listdir(self.augm_data_plumes_path)
            for filename in augm_data_plume_files:
                key = self._extract_key(filename, augm=True)
                plume_dict[key] = os.path.join(
                    self.augm_data_plumes_path, filename
                )

            augm_data_tile_files = os.listdir(self.augm_data_tiles_path)
            for filename in tqdm(
                augm_data_tile_files, desc="Mapping augmented tiles to plumes"
            ):
                key = self._extract_key(filename, augm=True)
                augm_data_tile_path = os.path.join(
                    self.augm_data_tiles_path, filename
                )

                augm_data_plume_path = plume_dict.get(key)
                if augm_data_plume_path:
                    mapping[augm_data_tile_path] = augm_data_plume_path
                else:
                    logger.info(
                        f"No augmented plume found for tile {augm_data_tile_path}"
                    )

        return mapping

    def __len__(self):
        # Return the size of the dataset
        return len(self.tile_files)

    def __getitem__(self, idx):
        # now for each idx in the list of tiles, extract the name,
        # from the dict the output name, and then load them with rasterio:
        tile_file = self.tile_files[idx]
        plume_file = self.mapping[tile_file]
        tile_data = np.load(tile_file, mmap_mode="r")
        if self.task == Task.SEM_SEGMENTATION:
            # Semantic Segmentation
            plume_data = np.load(plume_file, mmap_mode="r")
            concentration = plume_data
            # Convert ppm CH4 to 0,1 values
            plume_data = plume_data != -9999

            tile_data = torch.from_numpy(tile_data).clone()
            plume_data = torch.from_numpy(plume_data).clone()

            tile_data = torch.nan_to_num(tile_data, nan=-9999)

            if self.augm_data:
                if len(plume_data.shape) == 2:
                    plume_data = plume_data.unsqueeze(axis=0)
            # Return a single data item based on the index
            return (
                tile_data,
                plume_data,
                concentration,
                tile_file,
            )
        else:
            raise Exception("The defined task does not exist.")
