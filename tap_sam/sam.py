import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor


class Sam:
    def __init__(
        self,
        sam2_checkpoint,
        model_cfg,
        threshold=0.0,
        save_visualization=False,
        device="cpu",
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.threshold = threshold
        self.save_visualization = save_visualization
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, device
        )

    def get_mask_on_image(
        self, masks_list, video, obj_id=None, random_color=True, save_path=None
    ):
        if self.save_visualization:
            assert (
                save_path is not None
            ), "while save_visualizationis True, save_path must be provided."
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 2 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3]])

        mask_image = [torch.tensor(masks_list[i]).permute(2, 3, 1, 0).numpy() * (
            color.reshape(1, 1, -1)[:, :, :, None]
        ) for i in range(len(masks_list))]
        
        mask_image = [
            (torch.tensor(mask_image[i]).permute(3, 0, 1, 2) * 255)
            .numpy()
            .astype(np.uint8) for i in range(len(mask_image))
        ]
        mix_image_list = []

        # add mask to video
        width, height = mask_image[0][0].shape[1], mask_image[0][0].shape[0]
        if self.save_visualization:
            result = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height)
            )
            
        text_scale = width / 800
        assert video.shape[0] == mask_image[0].shape[0]
        for i in range(video.shape[0]):      
            for obj_id in range(len(masks_list)):
                mix_mask = masks_list[obj_id][i][0][:, :, None].repeat(3, axis=2)
                mix_image = np.where(mix_mask, mask_image[obj_id][i], video[i]) if obj_id == 0 else np.where(mix_mask, mask_image[obj_id][i], mix_image)
                # write number on the mask in the image by cv2
                loc = np.where(mix_mask[:,:,0])
                if len(loc[0]) == 0:
                    continue
                loc = (np.mean(loc[0]).astype(int), np.mean(loc[1]).astype(int))
                
                if loc[0] < 10:
                    loc = (10, loc[1])
                if loc[1] < 10:
                    loc = (loc[0], 10)
                if loc[0] > height - 10:
                    loc = (height - 10, loc[1])
                if loc[1] > width - 10:
                    loc = (loc[0], width - 10)
                
                cv2.putText(mix_image, str(obj_id+1), (loc[1], loc[0]), cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            mix_image_list.append(mix_image)
            if self.save_visualization:
                result.write(mix_image)
                result.release()

        return mix_image_list

    def set_video_path(self, video_path):
        self.video_path = video_path
        self.inference_state = self.predictor.init_state(video_path)
    
    def __call__(self, object_points, labels, select_frame, ann_obj_id):
        masks = []
        # inference_state = self.predictor.init_state(video_path)
        self.predictor.reset_state(self.inference_state)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=select_frame,
            obj_id=int(ann_obj_id),
            points=object_points,
            labels=labels,
        )
        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0)
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        masks = [
            video_segments[frame_idx][int(ann_obj_id)]
            for frame_idx in sorted(video_segments.keys())
        ]
        if masks[0].device != torch.device("cpu"):
            masks = [mask.cpu().numpy() for mask in masks]
        mask = np.stack(masks, axis=0)
        return mask
