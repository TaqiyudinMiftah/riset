import os
import subprocess
import shutil

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
# DOWNLOAD (SKIP IF EXISTS)
# =========================
print("📥 Checking & downloading files...")

for file_id, filename in FILES:
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"⏭️ Skip (already exists): {filename}")
        continue

    print(f"⬇️ Downloading: {filename}")
    cmd = [
        "gdown",
        f"https://drive.google.com/uc?id={file_id}",
        "-O",
        output_path,
        "--resume"
    ]
    subprocess.run(cmd, check=True)

print("✅ Download selesai!")

# =========================
# DETECT SPLIT ZIP
# =========================
split_files = [f for _, f in FILES if f.endswith(".z01")]

if split_files:
    print("\n🔍 Detected split zip archive")

    # =========================
    # CHECK 7z availability
    # =========================
    seven_zip_path = os.path.join(OUTPUT_DIR, "7zz")

    if not os.path.exists(seven_zip_path):
        print("📦 Downloading 7zip standalone...")

        subprocess.run(
            ["wget", "-q", "https://www.7-zip.org/a/7z2301-linux-x64.tar.xz"],
            cwd=OUTPUT_DIR,
            check=True
        )

        subprocess.run(
            ["tar", "-xf", "7z2301-linux-x64.tar.xz"],
            cwd=OUTPUT_DIR,
            check=True
        )

        print("✅ 7zip ready!")

    # =========================
    # EXTRACT WITH 7z
    # =========================
    print("\n📂 Extracting using 7z...")

    subprocess.run(
        ["./7zz", "x", "caers_split.zip"],
        cwd=OUTPUT_DIR,
        check=True
    )

else:
    print("\n📦 Single zip detected, using unzip...")

    subprocess.run(
        ["unzip", "caers_split.zip"],
        cwd=OUTPUT_DIR,
        check=True
    )

print("\n🎉 DONE! Dataset siap digunakan.")