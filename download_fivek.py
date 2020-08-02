import sys
import tqdm
import os
import glob
import subprocess

remote = "https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/"
fivek_dng_dir = "./fivek_dataset/raw_photos"
target_dir = "./fivek_dataset/expert_C"

os.makedirs(target_dir, exist_ok=True)

dngs = glob.glob(os.path.join(fivek_dng_dir, "*.dng"))
dngs = sorted(dngs)

print("Local dng files found:", len(dngs))

progress = tqdm.tqdm(range(len(dngs)), "Downloading fivek dataset", file=sys.stdout)
for i in progress:
    dng_file_name = os.path.basename(dngs[i])[:-4]

    tif_file_name = dng_file_name + ".tif"

    subprocess.call([
        "wget",
        "-q",
        "-P", target_dir,
        remote + tif_file_name
    ])

progress.close()

print("Done.")
