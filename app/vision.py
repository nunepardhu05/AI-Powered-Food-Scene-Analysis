"""CPU-friendly checkpoint adapter. No fake detections when artifacts are absent."""
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError
from .volume import resolve_overlaps
from .config import settings

MODEL_DIR=Path(__file__).resolve().parents[1]/'models'

def load_image(data):
    if len(data)>settings()['max_image_bytes']: raise ValueError('Image exceeds 10 MB')
    try:
        image=Image.open(io.BytesIO(data))
        if image.width*image.height>settings()['max_image_pixels']: raise ValueError('Image exceeds 16 megapixels')
        image.load()
        from PIL import ImageOps
        return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid or unsupported image') from exc

def quality(image):
    gray=np.asarray(image.convert('L'),float)
    edge=np.asarray(image.convert('L').filter(ImageFilter.FIND_EDGES),float)
    warnings=[]
    if gray.mean()<35: warnings.append('Dark image: retake in brighter light')
    if edge[2:-2,2:-2].var()<30 if min(gray.shape)>4 else True: warnings.append('Low-detail or blurry image: retake with the camera steady')
    return warnings

def analyze(image,reference_cm=None,reference_pixels=None):
    warnings=quality(image)
    if not reference_cm or not reference_pixels:
        warnings.append('Missing reference object. Default plate diameter 26 cm is an assumption, not a measured scale; no metric weight reported.')
    else:
        warnings.append('Reference gives image scale only; monocular depth still needs metric calibration.')
    if not (MODEL_DIR/'detector.pth').exists():
        return {'status':'models_unavailable','items':[],'warnings':warnings+['Install fine-tuned detector and recognition checkpoints. Confirm foods manually.'],'model_verified':False}
    pipeline=HybridPipeline()
    return pipeline.run(image,warnings)

class HybridPipeline:
    def __init__(self):
        import json
        import torch
        import timm
        from rfdetr import RFDETRBase, RFDETRSmall
        from transformers import AutoProcessor, Sam2Model, CLIPModel, CLIPProcessor, pipeline
        manifest=MODEL_DIR/'manifest.json'
        if not manifest.exists(): raise ValueError('models/manifest.json required with taxonomy and checkpoint configuration')
        self.meta=json.loads(manifest.read_text()); self.torch=torch
        self.device='cpu'
        constructor=RFDETRSmall if self.meta['detector_size']=='small' else RFDETRBase
        self.detector=constructor(pretrain_weights=str(MODEL_DIR/'detector.pth'),device='cpu')
        self.labels=self.meta['labels']
        self.candidates=list(dict.fromkeys(self.labels+self.meta.get('open_vocabulary',[])))
        self.classifier=timm.create_model(self.meta['backbone'],pretrained=False,num_classes=len(self.labels))
        self.classifier.load_state_dict(torch.load(MODEL_DIR/'classifier.pt',map_location='cpu',weights_only=True)['model'])
        self.classifier.eval()
        from timm.data import resolve_model_data_config, create_transform
        self.transform=create_transform(**resolve_model_data_config(self.classifier),is_training=False)
        # Artifacts must be downloaded explicitly; inference remains local.
        self.sam=Sam2Model.from_pretrained(str(MODEL_DIR/'sam2'),local_files_only=True).eval()
        self.sam_processor=AutoProcessor.from_pretrained(str(MODEL_DIR/'sam2'),local_files_only=True)
        self.clip=CLIPModel.from_pretrained(str(MODEL_DIR/'clip'),local_files_only=True).eval()
        self.clip_processor=CLIPProcessor.from_pretrained(str(MODEL_DIR/'clip'),local_files_only=True)
        self.depth=pipeline('depth-estimation',model=str(MODEL_DIR/'depth'),device=-1)

    def run(self,image,warnings):
        torch=self.torch; detections=self.detector.predict(image,threshold=.25)
        if len(detections.xyxy)==0: return {'status':'no_food_detected','items':[],'warnings':warnings,'model_verified':False}
        boxes=detections.xyxy.tolist()
        inputs=self.sam_processor(images=image,input_boxes=[boxes],return_tensors='pt')
        with torch.inference_mode(): outputs=self.sam(**{k:v for k,v in inputs.items() if k in ('pixel_values','input_boxes','input_points','input_labels')},multimask_output=False)
        masks=self.sam_processor.post_process_masks(outputs.pred_masks,inputs['original_sizes'])[0]
        masks=masks[:,0].cpu().numpy()>0
        masks,overlaps=resolve_overlaps(masks,detections.confidence)
        with torch.inference_mode(): relative_depth=np.asarray(self.depth(image)['depth'],float)
        items=[]
        for i,box in enumerate(boxes):
            crop=image.crop(tuple(box))
            with torch.inference_mode():
                logits=self.classifier(self.transform(crop).unsqueeze(0))[0]
                classifier=torch.softmax(logits,dim=-1).numpy()
                clip_inputs=self.clip_processor(text=[f'a photo of {x}' for x in self.candidates],images=crop,return_tensors='pt',padding=True)
                clip=torch.softmax(self.clip(**clip_inputs).logits_per_image[0],dim=-1).numpy()
            detector=np.zeros(len(self.candidates)); cid=int(detections.class_id[i])
            if 0<=cid<len(self.labels): detector[cid]=float(detections.confidence[i])
            classifier_scores=np.zeros(len(self.candidates));classifier_scores[:len(self.labels)]=classifier
            probabilities=.25*detector+.45*classifier_scores+.30*clip
            probabilities/=max(probabilities.sum(),1e-9)
            top=np.argsort(-probabilities)[:3]
            # Scores are deliberately not claimed to be calibrated probabilities.
            points=np.argwhere(masks[i]); step=max(1,len(points)//1500)
            items.append({'id':i,'bbox':box,'top3':[{'name':self.candidates[j],'score':float(probabilities[j])} for j in top], 'requires_confirmation':True,'overlap':overlaps[i],'mask_pixels_yx':points[::step].tolist(),'weight_g':None,'weight_range_g':None,'depth_mean_relative':float(relative_depth[masks[i]].mean()) if masks[i].any() else None,'assumptions':['Fusion scores uncalibrated','No metric depth calibration or validated weight regressor installed; enter a measured portion']})
        return {'status':'confirmation_required','items':items,'warnings':warnings,'model_verified':False}
