import requests
import re
import urllib3
urllib3.disable_warnings()

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for year in ['2007', '2009', '2011']:
    url = f'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear={year}'
    try:
        r = requests.get(url, verify=False, headers=headers, timeout=15)
        matches = re.findall(r'href=["\']([^"\']+\.XPT)["\']', r.text, re.IGNORECASE)
        for m in matches:
            if 'spx' in m.lower():
                print(f"Year {year} SPX match: {m}")
    except Exception as e:
        print(f"Year {year} error:", e)
