"""Geometry utilities. Depth must be calibrated camera-axis metric depth in cm."""
import numpy as np

def resolve_overlaps(masks, scores):
    masks=np.asarray(masks,dtype=bool)
    if masks.ndim!=3 or len(masks)!=len(scores): raise ValueError('N masks and scores required')
    counts=masks.sum(axis=0); overlap=counts>1
    occupied=np.zeros(masks.shape[1:],bool); output=np.zeros_like(masks)
    for idx in np.argsort(-np.asarray(scores),kind='stable'):
        output[idx]=masks[idx]&~occupied
        occupied|=masks[idx]
    return output,[bool((mask&overlap).any()) for mask in masks]

def fit_plane(depth_cm, plate_mask, food_mask):
    depth=np.asarray(depth_cm,float)
    valid=np.asarray(plate_mask,bool)&~np.asarray(food_mask,bool)&np.isfinite(depth)
    if valid.sum()<30: raise ValueError('At least 30 non-food plate pixels required')
    y,x=np.indices(depth.shape)
    design=np.column_stack([x[valid],y[valid],np.ones(valid.sum())])
    values=depth[valid]
    coeff=np.linalg.lstsq(design,values,rcond=None)[0]
    # Iterative median-residual clipping removes background contamination.
    for _ in range(4):
        residual=values-design@coeff
        scale=max(float(np.median(abs(residual-np.median(residual)))*1.4826),1e-6)
        keep=abs(residual)<3*scale
        if keep.sum()<30: break
        coeff=np.linalg.lstsq(design[keep],values[keep],rcond=None)[0]
    return coeff[0]*x+coeff[1]*y+coeff[2]

def estimate_volume(depth_cm, mask, plate_mask, cm_per_pixel, calibrated=False):
    if not calibrated: raise ValueError('Relative depth is not metric. Calibrate depth offset and scale first.')
    if not np.isfinite(cm_per_pixel) or cm_per_pixel<=0: raise ValueError('Positive metric image scale required')
    depth=np.asarray(depth_cm,float); mask=np.asarray(mask,bool)
    if not mask.any(): return {'volume_ml':0.0,'assumptions':['Empty mask']}
    if depth.shape!=mask.shape or np.asarray(plate_mask).shape!=mask.shape or not np.isfinite(depth).all(): raise ValueError('Finite depth and equal-sized masks required')
    plane=fit_plane(depth,plate_mask,mask)
    height=np.maximum(0,plane-depth)
    volume=float(height[mask].sum()*cm_per_pixel**2)
    return {'volume_ml':volume,'assumptions':['Orthographic, near-overhead view; cm³ equals ml','No hidden food under visible surface; calibrated metric depth required','Perspective and tilted-plate corrections are not modeled']}

def physics_weight(volume_ml,density,packing,relative_uncertainty=.35):
    from .nutrition import interval
    if volume_ml<0 or density<=0 or not 0<packing<=1 or not 0<=relative_uncertainty<=1: raise ValueError('Invalid physical quantities')
    return interval(volume_ml*density*packing,relative_uncertainty)
