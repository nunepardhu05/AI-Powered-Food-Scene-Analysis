import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.vision import load_image,analyze

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--reference-cm',type=float);p.add_argument('--reference-pixels',type=float);a=p.parse_args()
    result=analyze(load_image(Path(a.image).read_bytes()),a.reference_cm,a.reference_pixels)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
