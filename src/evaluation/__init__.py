from evaluation.metrics import run_evaluation as run_evaluation

if __name__ == "__main__":
    from pathlib import Path
    from config import settings

    pdf_path = Path(settings.DEFAULT_PDF_PATH)
    if pdf_path.exists():
        run_evaluation(str(pdf_path))
    else:
        print(f"File not found: {pdf_path}")
