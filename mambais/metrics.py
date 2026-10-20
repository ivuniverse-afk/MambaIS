"""Class-aware instance AP for ScanNet-style evaluation."""
import numpy as np


def _ap(rec, prec):
    mrec=np.r_[0.,rec,1.]; mpre=np.r_[0.,prec,0.]
    for i in range(len(mpre)-2,-1,-1): mpre[i]=max(mpre[i],mpre[i+1])
    idx=np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1])


def average_precision(predictions, ground_truth, classes, thresholds=None):
    thresholds = np.r_[.25, np.arange(.5, 1., .05)] if thresholds is None else np.asarray(thresholds)
    result={}
    for threshold in thresholds:
        aps=[]
        for cls in range(classes):
            gt=[g for g in ground_truth if g["label"]==cls]; pred=sorted([p for p in predictions if p["label"]==cls],key=lambda x:-x["score"])
            used=set(); tp=[]; fp=[]
            for p in pred:
                candidates=[(i,g) for i,g in enumerate(gt) if g["scene_id"]==p["scene_id"] and i not in used]
                ious=[(np.logical_and(p["mask"],g["mask"]).sum()/max(1,np.logical_or(p["mask"],g["mask"]).sum()),i) for i,g in candidates]
                best=max(ious,default=(0,-1)); ok=best[0]>=threshold
                tp.append(ok); fp.append(not ok)
                if ok: used.add(best[1])
            if gt:
                tp=np.cumsum(tp); fp=np.cumsum(fp); aps.append(_ap(tp/len(gt),tp/np.maximum(tp+fp,1)))
        result[float(threshold)]=float(np.mean(aps)) if aps else 0.
    map_values = [value for threshold, value in result.items() if threshold >= .5]
    return {"mAP":float(np.mean(map_values)),"AP50":result.get(.5,0.),
            "AP25":result.get(.25,0.),"per_threshold":result}
