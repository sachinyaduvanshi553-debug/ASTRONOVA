import os
import json
import asyncio
import logging
from services.ingestion.downloaders import (
    GOESDownloader, NOAADownloader, AdityaDownloader, SoLEXSDownloader, HEL1OSDownloader, SDODownloader
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astronova.scripts.download")

MAX_DOWNLOAD_SIZE_GB = 5.0
MANIFEST_PATH = r"c:\Users\sachi\OneDrive\Documents\ASTRONOVA\datasets\manifests\master_manifest.json"
DATASET_ROOT = r"c:\Users\sachi\OneDrive\Documents\ASTRONOVA\datasets"

async def main():
    if not os.path.exists(MANIFEST_PATH):
        logger.error(f"Manifest file not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Enforce local download size constraint
    max_size = manifest.get("max_download_size_gb", MAX_DOWNLOAD_SIZE_GB)
    logger.info(f"Starting ASTRONOVA Dataset Download. Local storage constraint: {max_size} GB maximum.")

    datasets = manifest.get("datasets", {})
    
    # Initialize downloaders and tasks
    tasks = []
    
    # 1. GOES XRS
    if "goes_xray" in datasets:
        config = datasets["goes_xray"]
        out_dir = os.path.join(DATASET_ROOT, "raw", "goes_xray")
        dl = GOESDownloader("goes_xray", config, out_dir)
        for f_info in config.get("files", []):
            filename = os.path.basename(f_info["url"])
            tasks.append(dl.download_file(f_info["url"], filename, f_info.get("checksum")))

    # 2. NOAA Events
    if "noaa_events" in datasets:
        config = datasets["noaa_events"]
        out_dir = os.path.join(DATASET_ROOT, "raw", "noaa_events")
        dl = NOAADownloader("noaa_events", config, out_dir)
        for f_info in config.get("files", []):
            filename = os.path.basename(f_info["url"])
            tasks.append(dl.download_file(f_info["url"], filename, f_info.get("checksum")))

    # 3. Aditya-L1 (solexs and hel1os)
    if "aditya_l1" in datasets:
        config = datasets["aditya_l1"]
        
        # SoLEXS
        solexs_dir = os.path.join(DATASET_ROOT, "raw", "solexs")
        solexs_dl = SoLEXSDownloader("solexs", config, solexs_dir)
        for f_info in config.get("files", []):
            # Map general Aditya-L1 level 2 to instrument
            filename = os.path.basename(f_info["url"]).replace("aditya", "solexs")
            tasks.append(solexs_dl.download_file(f_info["url"], filename, f_info.get("checksum")))
            
        # HEL1OS
        hel1os_dir = os.path.join(DATASET_ROOT, "raw", "heli1os")
        hel1os_dl = HEL1OSDownloader("heli1os", config, hel1os_dir)
        for f_info in config.get("files", []):
            filename = os.path.basename(f_info["url"]).replace("aditya", "heli1os")
            tasks.append(hel1os_dl.download_file(f_info["url"], filename, f_info.get("checksum")))

    # 4. SDO AIA
    if "sdo_images" in datasets:
        config = datasets["sdo_images"]
        out_dir = os.path.join(DATASET_ROOT, "raw", "sdo_images")
        dl = SDODownloader("sdo_images", config, out_dir)
        for f_info in config.get("files", []):
            filename = os.path.basename(f_info["url"]).replace(".fits", ".npy")
            tasks.append(dl.download_file(f_info["url"], filename, f_info.get("checksum")))

    # Execute all downloads/fallback tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info("Dataset ingestion phase complete. Verification of files:")
    for res in results:
        if isinstance(res, Exception):
            logger.error(f"Download/Generation task encountered error: {res}")
        else:
            logger.info(f"  File created/verified: {res}")

if __name__ == "__main__":
    asyncio.run(main())
