#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mambais.config import load_config
from mambais.loss import make_targets
from mambais.metrics import average_precision
from mambais.runtime import build,to_device


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description="Evaluate MambaIS instance AP")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",default="cuda"); p.add_argument("--score-threshold",type=float,default=.1)
    p.add_argument("--mask-threshold",type=float,default=.5); args=p.parse_args()
    cfg=load_config(args.config); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    model,loader=build(cfg,False); state=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(state.get("model",state)); model.to(device).eval(); predictions=[]; ground_truth=[]
    for batch in loader:
        batch=to_device(batch,device); out=model(batch)[0]; scene=batch["scene_id"][0]
        probs=out["pred_logits"].softmax(-1)[:,:-1]; scores,labels=probs.max(-1)
        masks=(out["pred_masks"].sigmoid()>args.mask_threshold)[:,out["point_to_voxel"]]
        for score,label,mask in zip(scores,labels,masks):
            if score>=args.score_threshold and mask.sum()>=1:
                predictions.append({"scene_id":scene,"score":float(score),"label":int(label),"mask":mask.cpu().numpy()})
        gt_labels,gt_masks=make_targets(batch["semantic"][0],batch["instance"][0],torch.arange(len(batch["instance"][0]),device=device),len(batch["instance"][0]))
        for label,mask in zip(gt_labels,gt_masks): ground_truth.append({"scene_id":scene,"label":int(label),"mask":mask.bool().cpu().numpy()})
    metrics=average_precision(predictions,ground_truth,len(cfg["data"]["class_names"]))
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()

