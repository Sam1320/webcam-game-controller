from tqdm import tqdm
import env
from utils import preprocessing

if __name__ == "__main__":
    test = True
    suffix = "_test" if test else ""
    src_folders = [f"{env.images_original_path}{suffix}/left", f"{env.images_original_path}{suffix}/right",
                   f"{env.images_original_path}{suffix}/wait"]
    dst_folders = [f"{env.images_processed_path}{suffix}/left", f"{env.images_processed_path}{suffix}/right",
                   f"{env.images_processed_path}{suffix}/wait"]

    print("processing images..")
    for src_folder, dst_folder in tqdm(zip(src_folders, dst_folders)):
        preprocessing.process_frames(src_folder, dst_folder)
    print("done")
