"""
Polyp Dataset Downloader & Extractor
====================================
AI-generated script for downloading and extracting polyp datasets.
Supports **resume download**, **auto-extraction**, and **cleanup**.

🔹 **Usage:**
    python download_dataset.py <dataset_name> | all

🔹 **Datasets:**
    - ETIS_LaribPolypDB
    - CVC_ColonDB
    - CVC_ClinicDB
    - Kvasir_SEG

# AI-generated: https://chatgpt.com/share/67df9a8c-d614-800f-af2a-a00c27fe6475
"""
import argparse
import os
import requests
import zipfile

# Dataset URLs
DATASETS = {
    "ETIS_LaribPolypDB": "https://nc.aseef.dev/s/ETIS_LaribPolypDB/download/ETIS-LaribPolypDB.zip",
    "CVC_ColonDB": "https://nc.aseef.dev/s/CVC_ColonDB/download/CVC-ColonDB.zip",
    "CVC_ClinicDB": "https://nc.aseef.dev/s/CVS_ClinicDB/download/CVC-ClinicDB.zip",
    "Kvasir_SEG": "https://nc.aseef.dev/s/Kvasir_SEG/download/Kvasir-SEG.zip",
    "LarynxDataset_Preliminary": "https://nc.aseef.dev/s/LarynxDataset_Preliminary/download/LarynxDataset-Preliminary.zip",
}


def download_file(url, output_path):
    """Download file with progress bar and resume support."""
    temp_file = output_path + ".part"
    headers = {}

    # Resume download if partial file exists
    if os.path.exists(temp_file):
        file_size = os.path.getsize(temp_file)
        headers["Range"] = f"bytes={file_size}-"
    else:
        file_size = 0

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code not in [200, 206]:
        print(f"Failed to download: {url} (HTTP {response.status_code})")
        return

    total_size = int(response.headers.get("content-length", 0)) + file_size
    print(f"Downloading: {output_path} ({total_size / (1024 * 1024):.2f} MB)")

    with open(temp_file, "ab") as file:
        downloaded = file_size
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.2f}%", end="", flush=True)

    # Rename only if the final file doesn't exist
    if not os.path.exists(output_path):
        os.rename(temp_file, output_path)

    # Cleanup: Remove temp file if fully downloaded
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"\nDownload complete: {output_path}")


def extract_zip(zip_path, extract_to):
    """Extract ZIP file if not already extracted."""

    print(f"Extracting: {zip_path} -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extraction complete: {extract_to}")

    # Cleanup: Delete ZIP file after successful extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"Deleted ZIP file: {zip_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract polyp datasets.")
    parser.add_argument("dataset", nargs="?", choices=list(DATASETS.keys()) + ["all"],
                        help="Specify which dataset to download or 'all' for all datasets.")

    args = parser.parse_args()

    if not args.dataset:
        parser.print_help()
        return

    datasets_to_download = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for _, url in datasets_to_download.items():
        zip_filename = url.split("/")[-1]

        # Step 1: Download
        if not os.path.exists(zip_filename):
            download_file(url, zip_filename)
        else:
            print(f"File already exists, skipping download: {zip_filename}")

        # Step 2: Extract
        extract_zip(zip_filename, ".")


if __name__ == "__main__":
    main()
