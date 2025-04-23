import os
import sys
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

# Add YOLOv5 directory to path
yolov5_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolov5-master'))
sys.path.append(str(yolov5_path))

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# Import our database module
from detection_database import DetectionDatabase

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=yolov5_path / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=yolov5_path / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=yolov5_path / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'), help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--save-to-db', action='store_true', help='save detection results to database')
    parser.add_argument('--analyze', action='store_true', help='analyze detection results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def run(opt):
    # Initialize database if needed
    db = None
    if opt.save_to_db:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'yolo_detections.db')
        db = DetectionDatabase(db_path)
    
    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    stride, names, pt = model.stride, model.names, model.pt
    
    # Set image size
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    
    # Dataloader
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)
    
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))
    dt, seen = [0.0, 0.0, 0.0], 0
    
    # Create output directory
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Record detection run in database
    run_id = None
    if db:
        run_id = db.record_detection_run(
            model_name=opt.weights[0] if isinstance(opt.weights, list) else opt.weights,
            source_path=str(opt.source),
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            result_path=str(save_dir)
        )
    
    # Process detections
    all_results = []
    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=opt.augment, visualize=opt.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
            
            # Format results for database
            image_results = []
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # Save results (image with detections)
                    if opt.save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        
                    # Record detection for database
                    if db and run_id:
                        x_min, y_min, x_max, y_max = (
                            xyxy[0].item(), xyxy[1].item(),
                            xyxy[2].item(), xyxy[3].item()
                        )
                        
                        image_results.append({
                            'image_path': str(p),
                            'class_id': c,
                            'class_name': names[c],
                            'confidence': conf.item(),
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max
                        })
                        
            # Stream results
            im0 = annotator.result()
            
            # Save to database
            if db and run_id and image_results:
                all_results.extend(image_results)
                
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            
            # Save images and labels
            if not opt.nosave:
                cv2.imwrite(save_path, im0)
                
    # Save all results to database
    if db and run_id and all_results:
        db.record_detection_results(run_id, all_results)
        
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    # Analyze results if requested
    if opt.analyze and db:
        stats = db.get_statistics()
        print("\nDetection Statistics:")
        print(f"Total runs: {stats['run_count']}")
        print(f"Total detections: {stats['detection_count']}")
        print(f"Average confidence: {stats['avg_confidence']:.2f}")
        print("\nClass distribution:")
        for cls_name, count in stats['class_distribution']:
            print(f"  {cls_name}: {count}")
    
    return save_dir

if __name__ == "__main__":
    opt = parse_args()
    check_requirements(exclude=('tensorboard', 'thop'))
    run(opt)
