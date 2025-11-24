import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch, yaml, cv2, os, shutil
import numpy as np
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


# Utility function for resizing and letterboxing images
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class Yolo11Heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        self.device = torch.device(device)
        # Load YOLO11 model checkpoint
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names  # class names list
        csd = ckpt['model'].float().state_dict()  # FP32 state dictionary
        self.model = Model(cfg, ch=3, nc=len(model_names)).to(self.device)
        csd = intersect_dicts(csd, self.model.state_dict(), exclude=['anchor'])
        self.model.load_state_dict(csd, strict=False)
        self.model.eval()
        print(f'Transferred {len(csd)}/{len(self.model.state_dict())} items')

        self.model_names = model_names
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.backward_type = backward_type
        # Retrieve the target layer reference from the string (e.g., "model.model[9]")
        self.target_layers = [eval(layer, {"model": self.model})]
        # Use the specified CAM method (e.g., GradCAM)
        self.method = eval(method)
        self.colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted_scores, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], 0, 1)[indices[0]], \
            torch.transpose(boxes_[0], 0, 1)[indices[0]], \
            xywh2xyxy(torch.transpose(boxes_[0], 0, 1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, img_path, save_path):
        # Remove the save directory if it exists and then create it
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # Preprocess the image
        img = cv2.imread(img_path)
        img, _, _ = letterbox(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # Initialize GradCAM activations and gradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # Postprocess to obtain YOLO-style detection results
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])

        # Generate and save heatmaps for the top detections (based on ratio and confidence threshold)
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break

            self.model.zero_grad()
            # Backpropagate based on the selected backward type
            if self.backward_type in ('class', 'all'):
                score = post_result[i].max()
                score.backward(retain_graph=True)
            if self.backward_type in ('box', 'all'):
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # Accumulate gradients based on the chosen type
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = (grads.gradients[0] + grads.gradients[1] +
                             grads.gradients[2] + grads.gradients[3] + grads.gradients[4])

            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            # Optionally, draw bounding boxes and confidence if desired (currently commented out)
            # cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
            #                                  f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
            #                                  cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(os.path.join(save_path, f'{i}.png'))


def get_params():
    return {
        'weight': 'E:/DeepLearning/YOLO/YOLOv8/ultralytics-main/ultralytics-main/runs/detect/11n/weights/best.pt',  # Path to YOLO11 weights
        'cfg': 'ultralytics/cfg/models/11/yolo11.yaml',  # Config file (adapt if necessary for YOLO11)
        'device': 'cuda:0',
        'method': 'GradCAM',  # Options: GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[9]',  # Default layer (will be overwritten for analysis)
        'backward_type': 'all',  # 'class', 'box', or 'all'
        'conf_threshold': 0.00,
        'ratio': 0.02  # Ratio of top detections to process
    }


if __name__ == '__main__':
    # Define the list of layers (by their layer references) to analyze
    layer_list = ['model.model[0]']
    image_path = 'E:/DeepLearning/YOLO/YOLOv8/ultralytics-main/ultralytics-main/ImageSets_4_class/images/train/File_1.bmp'  # Input image path
    base_save_dir = 'G:/chip_sort_particles/heatmap_result/yolo11_heatmaps'
    os.makedirs(base_save_dir, exist_ok=True)

    # For each layer in the list, generate and save the corresponding heatmaps
    for layer in layer_list:
        params = get_params()
        params['layer'] = layer
        heatmap_model = Yolo11Heatmap(**params)
        save_dir = os.path.join(base_save_dir, layer.replace('[', '_').replace(']', ''))
        heatmap_model(image_path, save_dir)
