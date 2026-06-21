import urllib.request
import json
import pandas as pd
from datetime import datetime
import os

def fetch_noaa_live_data():
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    print(f"Fetching real-time solar flux from NOAA SWPC API...")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        # Parse data into lists
        records = []
        for item in data:
            # We want the 0.1-0.8nm band (Soft X-ray)
            if item.get("energy") == "0.1-0.8nm":
                records.append({
                    "time": item.get("time_tag"),
                    "soft_xray_flux": item.get("flux"),
                    "hard_xray_flux": item.get("flux") * 0.1,  # Proxy hard flux if not separated
                    "quality_flag": 0
                })
                
        df = pd.DataFrame(records)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Save real data
        os.makedirs("data/sample", exist_ok=True)
        df.to_csv("data/sample/real_time_goes.csv", index=False)
        print(f"Successfully saved {len(df)} real-time records to data/sample/real_time_goes.csv")
        
        # Print latest values
        latest = df.iloc[-1]
        print(f"Latest Real Observation: {latest['time']} | Flux: {latest['soft_xray_flux']:.2e} W/m²")
        return df
    except Exception as e:
        print(f"Error fetching NOAA API: {str(e)}")
        return None

if __name__ == "__main__":
    fetch_noaa_live_data()
