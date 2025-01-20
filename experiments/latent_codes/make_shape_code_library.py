import os
import numpy as np
import pickle

from crisp.utils.file_utils import safely_make_folders, id_generator

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
shp_code_save_folder = os.path.join(script_dir, "data/spe3r/train/spe3r_202410312058_AHHT6/raw_obj_codes.npy")
export_folder = os.path.join(script_dir, "shape_code_libraries/ycbv/202408241403_U6GD9")


if __name__ == "__main__":
    print("Generating shape code library.")

    safely_make_folders([export_folder])

    raw_obj_code_db = np.load(shp_code_save_folder, allow_pickle=True).item()
    shape_code_dict = {}
    for obj_name in raw_obj_code_db.keys():
        shape_code_dict[obj_name] = np.mean(np.array(raw_obj_code_db[obj_name]), axis=0)

    export_file_path = os.path.join(export_folder, "shape_code_lib.pkl")
    with open(export_file_path, "wb") as f:
        pickle.dump(shape_code_dict, f)

    print(f"Shape code library saved to {export_file_path}")
