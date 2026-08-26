import os


def generate_master_report():
    print("--- Generating Master System Report ---")

    report_files = [
        "reports/dataset_audit.md",
        "reports/features/feature_verification.md",
        "reports/training_verification.md",
        "reports/inference/inference_verification.md",
        "reports/scientific_validation.md",
        "reports/system_benchmark.md",
        "reports/production_readiness.md",
    ]

    master_content = "# ASTRONOVA V2 - Master Verification & Audit Report\n\n"
    master_content += "> Consolidated report of the 8-step end-to-end verification sprint.\n\n"

    for file in report_files:
        if os.path.exists(file):
            with open(file, encoding="utf-8") as f:
                content = f.read()
                # Remove title to make it a section
                content = content.replace("# ", "## ", 1)
                master_content += f"{content}\n\n---\n\n"
        else:
            master_content += f"## Missing Report: {file}\n\n---\n\n"

    with open("reports/master_system_report.md", "w", encoding="utf-8") as f:
        f.write(master_content)

    print("Master report generated at: reports/master_system_report.md")


if __name__ == "__main__":
    generate_master_report()
