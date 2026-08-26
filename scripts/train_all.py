import os
import subprocess


def main():
    models = ["xgboost", "lightgbm", "lstm"]

    print("========================================")
    print("ASTRONOVA ENSEMBLE TRAINING PIPELINE")
    print("========================================")

    for model in models:
        print(f"\n---> Starting training for {model.upper()}...")
        cmd = ["python", "ml/training/trainer.py", "--model", model]

        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"\n[ERROR] Training failed for {model}. Aborting pipeline.")
            return

        print(f"\n[SUCCESS] {model.upper()} training completed.")

    print("\n========================================")
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()
