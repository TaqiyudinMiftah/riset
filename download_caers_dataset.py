import os
import subprocess

# =========================
# CONFIG
# =========================
FILES = [
    ("1TGwKprghhECNyCRJzi7XhbH4O3n0N_Xf", "caers_split.z01"),
    ("1NqTGKG_9i6c0QmeTb4o2xKlAZSSr1QPA", "caers_split.z02"),
    ("1qt_C3lqn0MESRRkyp1JvNj23u9kWwrZW", "caers_split.z03"),
    ("1MHinoWJIU7EdL7Wt6tNHTM-1NtmJUFti", "caers_split.z04"),
    ("1fxKxU2YjvC9XLcOz4OcvU1_DMJP_kLUc", "caers_split.zip"),
]

OUTPUT_DIR = "caer_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DOWNLOAD
# =========================
print("📥 Downloading files...")
for file_id, filename in FILES:
    output_path = os.path.join(OUTPUT_DIR, filename)
    cmd = f"gdown https://drive.google.com/uc?id={file_id} -O {output_path}"
    subprocess.run(cmd, shell=True, check=True)

print("✅ Download selesai!")

# =========================
# MERGE SPLIT ZIP
# =========================
print("\n🔗 Merging split zip...")
subprocess.run(
    ["zip", "-s", "0", "caers_split.zip", "--out", "caers.zip"],
    cwd=OUTPUT_DIR,
    check=True
)

# =========================
# EXTRACT
# =========================
print("\n📂 Extracting dataset...")
subprocess.run(
    ["unzip", "caers.zip"],
    cwd=OUTPUT_DIR,
    check=True
)

print("\n🎉 DONE! Dataset siap digunakan.")


# #!/usr/bin/env python3
# """Download and extract the Balanced CAER-S dataset from Kaggle."""

# import os
# import sys
# import json
# import argparse
# from pathlib import Path

# from dotenv import load_dotenv


# def require_env(var_name: str) -> str:
#     value = os.getenv(var_name)
#     if not value:
#         print(f"Error: environment variable {var_name} is not set.")
#         sys.exit(1)
#     return value


# def ensure_kaggle_json(username: str, key: str) -> None:
#     kaggle_dir = Path.home() / ".kaggle"
#     kaggle_dir.mkdir(parents=True, exist_ok=True)
#     kaggle_json = kaggle_dir / "kaggle.json"
#     kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
#     kaggle_json.chmod(0o600)


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Download Balanced CAER-S dataset from Kaggle")
#     parser.add_argument(
#         "--force",
#         action="store_true",
#         help="Force re-download even if output directory is not empty",
#     )
#     args = parser.parse_args()

#     env_path = Path(__file__).resolve().parent / ".env"
#     load_dotenv(dotenv_path=env_path, override=True)

#     dataset = "dollyprajapati182/balanced-caer-s-dataset-7575-grayscale"
#     output_dir = Path("data/caers_balanced")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if not args.force and any(output_dir.iterdir()):
#         print(f"Skip download: {output_dir} already contains files.")
#         print("Use --force to re-download the dataset.")
#         return

#     # Validate credentials and create kaggle.json for API compatibility.
#     username = require_env("KAGGLE_USERNAME")
#     key = require_env("KAGGLE_KEY")
#     ensure_kaggle_json(username, key)
#     os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")

#     from kaggle.api.kaggle_api_extended import KaggleApi

#     api = KaggleApi()
#     api.authenticate()

#     print(f"Downloading {dataset} to {output_dir}...")
#     api.dataset_download_files(dataset=dataset, path=str(output_dir), unzip=True)
#     print("Download complete.")


# if __name__ == "__main__":
#     main()
