# Copyright (c) Gorilla Lab, SCUT.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation of results of DualPoseNet on CAMERA25 and REAL275.
"""

import datetime
import random
import string
import os

from crisp.datasets.nocs_evaluate import evaluate
from crisp.utils.file_utils import safely_make_folders, id_generator


#CAMERA25_path = os.path.join("results", "CAMERA25")
#REAL275_path = os.path.join("results", "REAL275")
# umeyama
#REAL275_path = "/home/jnshi/code/Hydra-Objects/experiments/nocs_ssl/model_ckpts/202411081606_1UUGC"
## arun
#REAL275_path = "/home/jnshi/code/Hydra-Objects/experiments/nocs_ssl/model_ckpts/202411081638_Y61LK"
# corrector -> umeyama
#REAL275_path = "/home/jnshi/code/Hydra-Objects/experiments/nocs_ssl/model_ckpts/202411082122_IP5ZW"
# camera sl -> umeyama
REAL275_path = "/home/jnshi/code/Hydra-Objects/experiments/nocs_ssl/model_ckpts/202411090928_XK669"


if __name__ == "__main__":
    # print('\n*********** Evaluate the results of DualPoseNet on CAMERA25 ***********')
    # evaluate(CAMERA25_path)

    print("\n*********** Evaluate the results of DualPoseNet on REAL275  ***********")
    exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"
    artifacts_save_dir = os.path.join("artifacts", exp_id)
    print(f"Artifacts save dir: {artifacts_save_dir}")
    safely_make_folders([artifacts_save_dir])
    evaluate(REAL275_path, artifacts_save_dir=artifacts_save_dir)
