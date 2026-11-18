from pathlib import Path
from pipeline import process_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Student_folder" / "Student folder"
OUTPUT = ROOT / "results" / "processed"

print("Starting PathSnap data processing...")
print("Dataset:", DATA)
print("Output:", OUTPUT)

if not (DATA / "train").is_dir():
    raise FileNotFoundError(f"Training data not found: {DATA / 'train'}")

if not (DATA / "student_test").is_dir():
    raise FileNotFoundError(f"Student test data not found: {DATA / 'student_test'}")

summary = process_dataset(DATA, OUTPUT)

print("\n========== PROCESSING SUMMARY ==========")
print(summary.groupby("partition")[[
    "candidate_rows",
    "fit_rows",
    "positive_fit_rows",
    "usable_label_observations",
    "reference_candidate_misses"
]].sum())

print("\n========== OUTPUT VERIFICATION ==========")
for partition in ("train", "student_test"):
    folders = sorted((OUTPUT / partition).glob("*"))
    valid = sum(
        (folder / "candidates.npz").exists()
        and (folder / "targets.npz").exists()
        for folder in folders
    )
    print(f"{partition}: {valid} trips with candidate and target files")

print("\nProcessing complete.")
