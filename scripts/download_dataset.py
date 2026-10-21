"""Download licensed data without bundling or redistributing third-party datasets."""
import argparse
import hashlib
from pathlib import Path
import httpx

SOURCES={
 'food101':('https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz','https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/'),
 'nutrition5k':(None,'https://github.com/google-research-datasets/Nutrition5k'),
 'uecfood256':(None,'https://foodcam.mobi/dataset256.html'),
 'foodseg103':(None,'https://github.com/LARC-CMU-SMU/FoodSeg103'),
 'regional':(None,'Supply a licensed regional dataset URL and record its terms in your manifest')}

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SOURCES);p.add_argument('--url');p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--accept-license',action='store_true');a=p.parse_args()
    default,license_url=SOURCES[a.dataset]
    if not a.accept_license:raise SystemExit(f'Review the license at {license_url}, then pass --accept-license')
    url=a.url or default
    if not url or not url.startswith('https://'):raise SystemExit('Supply the current HTTPS download URL from the official dataset page')
    if len(a.sha256)!=64:raise SystemExit('Provide the published or independently verified SHA-256 checksum')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);partial=out.with_suffix(out.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
    with httpx.stream('GET',url,headers={'Range':f'bytes={offset}-'} if offset else {},follow_redirects=True,timeout=120) as response:
        response.raise_for_status()
        append=offset>0 and response.status_code==206
        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Unexpected resume response')
        with partial.open('ab' if append else 'wb') as f:
            for chunk in response.iter_bytes():f.write(chunk)
    digest=hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()
    if digest!=a.sha256:raise ValueError('Checksum mismatch; partial retained for inspection')
    partial.replace(out);print(f'Verified download: {out}')
if __name__=='__main__':main()
