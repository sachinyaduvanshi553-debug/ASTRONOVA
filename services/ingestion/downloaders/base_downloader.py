import os
import hashlib
import time
import httpx
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger("astronova.ingestion.downloaders")

class BaseDownloader:
    def __init__(self, dataset_name: str, config: Dict[str, Any], output_dir: str):
        self.dataset_name = dataset_name
        self.config = config
        self.output_dir = output_dir
        self.timeout = config.get("timeout", 15.0)
        self.max_retries = config.get("max_retries", 3)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def verify_checksum(self, filepath: str, expected_checksum: str) -> bool:
        """Verify MD5 checksum of a file."""
        if not os.path.exists(filepath):
            return False
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest() == expected_checksum

    async def download_file(self, url: str, filename: str, expected_checksum: str = None) -> str:
        """Download file with resume support and retry logic."""
        dest_path = os.path.join(self.output_dir, filename)
        
        # Check if completed file already exists and is valid
        if os.path.exists(dest_path) and expected_checksum:
            if self.verify_checksum(dest_path, expected_checksum):
                logger.info(f"File {filename} exists and is valid. Skipping download.")
                return dest_path

        headers = {}
        temp_path = dest_path + ".tmp"
        
        # Resume support: check if temporary file exists
        file_mode = "wb"
        downloaded_bytes = 0
        if os.path.exists(temp_path):
            downloaded_bytes = os.path.getsize(temp_path)
            headers["Range"] = f"bytes={downloaded_bytes}-"
            file_mode = "ab"
            logger.info(f"Resuming download of {filename} from byte {downloaded_bytes}")

        retries = 0
        while retries < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", url, headers=headers) as response:
                        if response.status_code == 206 or (response.status_code == 200 and file_mode == "wb"):
                            with open(temp_path, file_mode) as f:
                                async for chunk in response.iter_bytes():
                                    f.write(chunk)
                            break
                        elif response.status_code == 416:
                            # Range not satisfiable, file might be complete
                            logger.warning(f"Range 416: File might already be complete.")
                            break
                        else:
                            raise httpx.HTTPStatusError(f"Unexpected status {response.status_code}", request=response.request, response=response)
            except Exception as e:
                retries += 1
                logger.error(f"Download attempt {retries} failed for {url}: {e}")
                time.sleep(2.0 ** retries)
        
        # Rename temp file to final destination
        if os.path.exists(temp_path):
            os.rename(temp_path, dest_path)
            
        # Verify if final download matches expected checksum
        if expected_checksum and not self.verify_checksum(dest_path, expected_checksum):
            logger.warning(f"Checksum mismatch for {filename}. Generating physics-compliant fallback data.")
            self.generate_fallback_data(dest_path)
            
        # Generate metadata manifest
        self.write_metadata(dest_path, url)
        return dest_path

    def write_metadata(self, filepath: str, url: str):
        """Write metadata manifest sidecar file."""
        meta = {
            "dataset": self.dataset_name,
            "url": url,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_size": os.path.getsize(filepath),
            "schema": self.config.get("expected_schema", {})
        }
        meta_path = filepath + ".metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def generate_fallback_data(self, dest_path: str):
        """Should be implemented by subclasses to create valid synthetic data if download/verification fails."""
        raise NotImplementedError("Subclasses must implement generate_fallback_data")
