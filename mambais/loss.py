"""Hungarian set matching and the MambaIS objective (Eqs. 14-17)."""

import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def make_targets(semantic, instance, point_to_voxel, num_voxels):
    labels, masks = [], []
    for inst in torch.unique(instance):
        if inst < 0: continue
        point_mask = instance == inst
        valid = semantic[point_mask]
        valid = valid[valid >= 0]
        if not len(valid): continue
        label = torch.mode(valid).values
        mask = torch.zeros(num_voxels, device=instance.device)
        mask[point_to_voxel[point_mask].unique()] = 1
        labels.append(label); masks.append(mask)
    return (torch.stack(labels) if labels else instance.new_empty(0),
            torch.stack(masks) if masks else torch.empty(0, num_voxels, device=instance.device))


def dice_loss(logits, targets):
    probs = logits.sigmoid()
    return 1 - (2*(probs*targets).sum(-1)+1) / (probs.sum(-1)+targets.sum(-1)+1)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, class_weight=.5, bce_weight=1., dice_weight=1., no_object=.1, auxiliary=True):
        super().__init__()
        self.num_classes = num_classes
        self.weights = (class_weight, bce_weight, dice_weight)
        self.no_object, self.auxiliary = no_object, auxiliary

    @torch.no_grad()
    def match(self, logits, masks, labels, targets):
        if not len(labels): return [], []
        cls = -logits.softmax(-1)[:, labels]
        bce = torch.stack([F.binary_cross_entropy_with_logits(masks, t.expand_as(masks), reduction="none").mean(1) for t in targets], 1)
        dice = torch.stack([dice_loss(masks, t.expand_as(masks)) for t in targets], 1)
        rows, cols = linear_sum_assignment((self.weights[0]*cls+self.weights[1]*bce+self.weights[2]*dice).detach().cpu())
        return torch.as_tensor(rows, device=logits.device), torch.as_tensor(cols, device=logits.device)

    def one(self, output, labels, masks):
        rows, cols = self.match(output["pred_logits"], output["pred_masks"], labels, masks)
        target_cls = torch.full((len(output["pred_logits"]),), self.num_classes, device=labels.device, dtype=torch.long)
        if len(rows): target_cls[rows] = labels[cols]
        weight = torch.ones(self.num_classes+1, device=labels.device); weight[-1] = self.no_object
        ce = F.cross_entropy(output["pred_logits"], target_cls, weight=weight)
        if len(rows):
            pred, target = output["pred_masks"][rows], masks[cols]
            bce = F.binary_cross_entropy_with_logits(pred, target)
            dice = dice_loss(pred, target).mean()
        else:
            bce = output["pred_masks"].sum()*0; dice = bce
        return self.weights[0]*ce + self.weights[1]*bce + self.weights[2]*dice, {"loss_ce":ce,"loss_bce":bce,"loss_dice":dice}

    def forward(self, outputs, batch):
        total = outputs[0]["pred_masks"].sum()*0; stats = {"loss_ce":0.,"loss_bce":0.,"loss_dice":0.}
        for out, sem, inst in zip(outputs, batch["semantic"], batch["instance"]):
            labels, masks = make_targets(sem, inst, out["point_to_voxel"], out["pred_masks"].shape[1])
            loss, parts = self.one(out, labels, masks); total = total + loss
            for key in stats: stats[key] = stats[key] + parts[key].detach()
            if self.auxiliary:
                for aux in out["aux_outputs"]:
                    aux_loss, _ = self.one(aux, labels, masks); total = total + .25*aux_loss
        n=max(1,len(outputs)); return total/n, {k: float(v/n) for k,v in stats.items()}

