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
        self.ann_obj_id = 1
        self.threshold = threshold
        self.save_visualization = save_visualization
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, device
        )

    def get_mask_on_image(
        self, masks, video, obj_id=None, random_color=False, save_path=None
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

        mask_image = torch.tensor(masks).permute(2, 3, 1, 0).numpy() * (
            color.reshape(1, 1, -1)[:, :, :, None]
        )
        mask_image = (
            (torch.tensor(mask_image).permute(3, 0, 1, 2) * 255)
            .numpy()
            .astype(np.uint8)
        )
        mix_image_list = []

        # add mask to video

        width, height = mask_image[0].shape[1], mask_image[0].shape[0]
        if self.save_visualization:
            result = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height)
            )
        for i in range(len(mask_image)):
            mix_mask = masks[i][0][:, :, None].repeat(3, axis=2)
            mix_image = np.where(mix_mask, mask_image[i], video[i])
            mix_image_list.append(mix_image)
            if self.save_visualization:
                result.write(mix_image)
                result.release()

        return mix_image_list, masks

    def __call__(self, video_path, object_points, labels, select_frame):
        masks = []
        inference_state = self.predictor.init_state(video_path)
        self.predictor.reset_state(inference_state)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=select_frame,
            obj_id=self.ann_obj_id,
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
        ) in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0)
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        masks = [
            video_segments[frame_idx][self.ann_obj_id]
            for frame_idx in sorted(video_segments.keys())
        ]
        if masks[0].device != torch.device("cpu"):
            masks = [mask.cpu().numpy() for mask in masks]
        mask = np.stack(masks, axis=0)
        return mask
