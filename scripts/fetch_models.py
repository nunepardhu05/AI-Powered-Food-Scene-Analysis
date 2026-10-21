"""Explicit download only; review model licenses before accepting."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--accept-model-licenses',action='store_true');a=p.parse_args()
    if not a.accept_model_licenses:raise SystemExit('Review SAM2, OpenAI CLIP, Depth Anything V2 and RF-DETR model licenses before downloading')
    from transformers import AutoModelForDepthEstimation,AutoImageProcessor,Sam2Model,AutoProcessor,CLIPModel,CLIPProcessor
    root=Path(__file__).resolve().parents[1]/'models'
    sources=json.loads((root.parent/'data/model-sources.json').read_text())
    for model_class,processor_class,folder in [(Sam2Model,AutoProcessor,'sam2'),(CLIPModel,CLIPProcessor,'clip'),(AutoModelForDepthEstimation,AutoImageProcessor,'depth')]:
        repo=sources[folder]['repository'];revision=sources[folder]['revision'];target=root/folder
        model_class.from_pretrained(repo,revision=revision).save_pretrained(target)
        processor_class.from_pretrained(repo,revision=revision).save_pretrained(target)
        (target/'source_revision.json').write_text(json.dumps(sources[folder],indent=2))
        print(f'Downloaded {repo}@{revision} to {target}')
if __name__=='__main__':main()
