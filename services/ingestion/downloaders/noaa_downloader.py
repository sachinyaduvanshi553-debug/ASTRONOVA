import os
from datetime import datetime
from services.ingestion.downloaders.base_downloader import BaseDownloader

class NOAADownloader(BaseDownloader):
    def generate_fallback_data(self, dest_path: str):
        """Generate typical NOAA SWPC solar event reports text file."""
        content = """:Product: 20260621events.txt
:Created: 2026 Jun 22 0210 UTC
# Prepared by the Joint USAF/NOAA Space Weather Operations.
#
# NOAA/USAF Space Weather Event List
#
# Event      Start Max   End   Obs  Q  Type  Loc/Frq   Class  Reg#
#-----------------------------------------------------------------
9800 +       0200  0220  0240  LEA  3  FLA   S15W20    C8.5   3420
9801         0600  0615  0700  KAN  3  FLA   N10E45    M4.2   3421
9802 +       1000  1010  1130  GONG 4  FLA   S20W10    X1.5   3422
9803         1230  1245  1315  LEA  3  RSP   024-048   III    3420
9804 +       1500  1510  1530  LEA  3  FLA   N10E45    M1.0   3421
"""
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
