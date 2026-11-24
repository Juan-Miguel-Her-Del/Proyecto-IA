# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.
import numpy as np

def box_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0.

def nms(boxes,scores,threshold=.45,max_detections=20):
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    scores=np.asarray(scores)
    order=np.argsort(-scores,kind='stable')
    areas=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    keep=[]
    while len(order) and len(keep)<max_detections:
        i=int(order[0]);keep.append(i)
        rest=order[1:]
        intersection=np.maximum(0,np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2])).prod(axis=1)
        overlap=intersection/np.maximum(areas[i]+areas[rest]-intersection,1e-9)
        order=rest[overlap<=threshold]
    return keep

def fuse_detection_views(views, association_iou=.50, lone_factor=.78, final_nms=.45, min_confidence=0.):
    """Refine boxes that agree across views and lower isolated detection scores."""
    active = [(source, prediction) for source, view in enumerate(views) for prediction in view]
    if not active:
        return []
    boxes = np.array([prediction['box'] for _, prediction in active], dtype=np.float32)
    scores = np.array([prediction['score'] for _, prediction in active])
    sources = np.array([source for source, _ in active])
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    remaining = np.ones(len(active), dtype=bool)
    fused, confidence = [], []
    for seed in np.argsort(-scores, kind='stable'):
        if scores[seed] < min_confidence:
            break  # The fused confidence cannot exceed the maximum input score.
        if not remaining[seed]:
            continue
        overlap = np.maximum(0, np.minimum(boxes[seed, 2:], boxes[:, 2:]) - np.maximum(boxes[seed, :2], boxes[:, :2])).prod(1)
        iou = overlap / np.maximum(areas[seed] + areas - overlap, 1e-9)
        matching = remaining & (iou >= association_iou)
        partners = [seed]
        for source in range(len(views)):
            if source == sources[seed]:
                continue
            choices = np.flatnonzero(matching & (sources == source))
            if len(choices):
                partners.append(choices[np.argmax(scores[choices] * iou[choices])])
        remaining[partners] = False
        weights = scores[partners]
        fused.append(np.average(boxes[partners], axis=0, weights=weights).tolist())
        score = (0.6 * weights.max() + 0.4 * weights.mean()) if len(partners) > 1 else weights[0] * lone_factor
        confidence.append(float(score))
    return [{'box': fused[i], 'score': confidence[i]} for i in nms(fused, confidence, final_nms, max_detections=100)
            if confidence[i] >= min_confidence]
