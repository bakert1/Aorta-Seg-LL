import gc
import os
from typing import Optional, Tuple

import monai.inferers
import monai.transforms
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from scipy import ndimage
import torch

import constants
import network
import utils


class UNetFastModule(pl.LightningModule):
    def __init__(
            self,
            do_seg: bool = True,
            do_ll: bool = True,
            n_seg_classes: int = 1,
            n_landmarks: int = 6,
            in_channels: int = 1,
            kernel_size: Tuple[int, int, int] = (3, 3, 3),
            channel_size_scale: int = 1,
            drop_rate: float = 0.0,
            seg_loss: str = 'bce',
            hmap_loss: str = 'wbce',
            hmap_wbce_loss_weight: float = 0.9,
            lmbda: float = 10.,
            lr: float = 1e-4,
            sw_roi_size: Tuple[int, int, int] = constants.DEFAULT_CROP_SIZE,
            sw_batch_size: int = 4,
            sw_overlap: float = 0.50,
            sw_mode: str = "gaussian",
            seg_postfix: str = "UNet_VOI",
            ll_postfix: str = "UNet_landmarks",
            pep_postfix: str = "UNet_post_eval_pipe",
            pred_debug: bool = False,
            pred_redo: bool = False,
            pred_verbose: bool = True,
         ):
        """
        See PyTorch Lightning documentation for information about how LightningModules work and for more information
        about how training_step, validation_step, etc. are used.

        Args:
            do_seg:
                Whether to use and process segmentation data. Set to False if no segmentation label is available.
            do_ll:
                Whether to use and process ll data. Set to False if no landmark labels are available.
            n_seg_classes:
                Number of segmentation classes. Currently, only 0 or 1 segmentation class is supported.
            n_landmarks:
                Number of landmarks.
            in_channels:
                Number of input channels. Only 1 input_channel is currently supported (i.e., only gray-scale images are
                supported).
            kernel_size:
                Size of U-Net kernel.
            channel_size_scale:
                Multiplier for the number of channels used in the U-Net. Increasing this parameter makes the U-Net
                bigger in terms of layer size, but not in terms of depth. Note that using channel_size_scale > 1 has
                not been extensively tested.
            drop_rate:
                Probability used for the U-Net's dropout layers.
            seg_loss:
                Loss function for segmentation.
            hmap_loss:
                Loss function for heatmap regression.
            hmap_wbce_loss_weight:
                Specifies the weight of the heatmap's binary cross entropy loss (WBCE). Only used when hmap_loss="wbce".
            lmbda:
                Specifies "lambda" the relative weighting of segmentation and landmark localization loss. The total loss
                function is: total_loss = seg_loss + lambda * hmap_loss.
            lr:
                Learning rate used in optimizer.
            sw_roi_size:
                Sliding window (SW) region of interest (ROI) size. Sliding window inference is used because medical
                images are usually too large to fit on GPU at once. It is also helpful to use overlapping sliding
                windows to improve segmentation performance because segmentation tends to be worse near edge of
                window.
            sw_batch_size:
                Batch size for sliding window inference. Increasing this speeds up inference, but requires more GPU
                memory. It does not affect accuracy.
            sw_overlap:
                How much to overlap the windows during sliding window inference.
            sw_mode:
                How to combine overlapping windows during sliding window inference.
            seg_postfix:
                Postix term added to image filename when creating a segmentation prediction. For example, if CT is named
                "image.nii.gz" then segmentation prediction will be named "image_{seg_postfix}.nii.gz". This parameter
                is only used when network is in prediction mode and not when it is in fit, validate, or testing mode.
            ll_postfix:
                Postix term added to image filename when creating a landmark prediction. See also seg_postfix
                description.
            pep_postfix:
                Postfix term added to image filename when creating a post-eval-pipe image. See also seg_postfix
                description. This parameter is only used when pred_debug=True.
            pred_debug:
                If True, saves extra data when predicting segmentation and landmarks for debugging purposes. Only used
                when network is in predict mode.
            pred_redo:
                If True, prediction input images are not skipped if a prediction file already exists.
            pred_verbose:
                If True, the file names of already completed predictions that are skipped are each printed out.
                Otherwise, only the number of skipped files are printed. Generally, setting this to False is helpful if
                your prediction list has 1000s of scans and many are already complete (printing each completed scan can
                be slow). This variable has no affect is pred_redo=True because no files are skipped in that case.
        """
        assert n_seg_classes == 1, "Current implementation only supports case of a single segmentation class."
        assert in_channels == 1, "The case of in_channels > 1 has not been tested."

        super().__init__()
        self.unet = network.UNet(do_seg, do_ll, n_seg_classes, n_landmarks, in_channels, channel_size_scale,
                                 kernel_size, drop_rate)

        self.seg_criterion = utils.get_seg_criterion(seg_loss)
        self.hmap_criterion = utils.get_heatmap_criterion(hmap_loss, hmap_wbce_loss_weight)
        self.lmbda = lmbda if do_seg else 1
        self.lr = lr
        self.sw_roi_rize = sw_roi_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap
        self.sw_mode = sw_mode
        self.seg_postfix = seg_postfix
        self.ll_postfix = ll_postfix
        self.pep_postfix = pep_postfix
        self.pred_debug = pred_debug
        self.pred_redo = pred_redo
        self.pred_verbose = pred_verbose

        self.boundary_finder = utils.BoundaryFromSeg()
        self.skipped_preds = []

        self.test_seg_df = pd.DataFrame(
            columns=["Seg_Type", "Dice", "IOU", "Hausdorff", "Hausdorff95", "Hausdorff99", "ASSD"]
        )
        self.test_ll_df = pd.DataFrame(columns=["Landmark_Num", "Landmark", "MSE_Loss", "WBCE_Loss", "Distance"])

        self.save_hyperparameters()

    def forward(self, x):
        """This function is called by the default predict_step. We override predict_step so this function shouldn't be
        used.
        """
        pass

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: the current batch of training data
            batch_idx: the index of the current batch

        Returns: the training loss

        """
        # R: number of random crops; N: training batch size; H,W,D: crop height, width, and depth;
        # L: number of landmarks
        x = batch["img"]  # Input CT. Shape: (R*N, 1, H, W, D)

        # run network
        seg_logits, hmap = self.unet(x)  # always returns (seg_logits, hmap) but seg_logits and hmap can be None.

        log_data = {}
        total_loss = 0
        if self.unet.do_seg:
            y = batch["seg"]  # CT seg label. Shape: (R*N, 1, H, W, D)
            y = torch.clip(y, min=0.0, max=1.0)  # Clip shouldn't be needed, but is included just in case of bad data.
            seg_loss = self.seg_criterion(input=seg_logits, target=y)  # Shape: (1,)
            log_data["train/seg_loss"] = seg_loss

            total_loss += seg_loss

        if self.unet.do_ll:
            z = batch["ll"]  # CT landmark localization label. Shape: (R*N, L, H, W, D)
            hmap_loss = self.hmap_criterion(input=hmap, target=z)  # Shape: (R*N, L, H, W, D)
            hmap_loss = hmap_loss.mean(dim=(0, -1, -2, -3))  # Avg over batch and spatial dims. Shape: (L,)

            for idx in range(self.unet.n_landmarks):
                log_data[f"train/hmap_losses/L{idx}"] = hmap_loss[idx]
            log_data["train/hmap_loss"] = hmap_loss.mean()

            total_loss += self.lmbda * hmap_loss.mean()

        # Logging
        log_data["train/total_loss"] = total_loss
        batch_size = x.shape[0]
        self.log_dict(log_data, on_epoch=True, on_step=False, batch_size=batch_size, sync_dist=True)

        return total_loss

    def _shared_val_test_step(self, batch, prefix):
        """Defines processing shared by both validation and testing.

        The loss is logged alongside other metrics like Dice/ASSD/Hausdorff for segmentation and L2 distance for
        landmark localization.

        Additionally, the segmentation and landmark predictions are returned for further processing by validation_step
        and test_step.

        Args:
            batch: technically a "batch" of input data, but batch size is always 1 for evaluation
            prefix: either "val" or "test" depending on whether validation_step or test_step called this function.
            prefix is used to label the logging appropriately.

        Returns: a dictionary of network outputs. If self.unet.do_seg, network_outputs includes the "seg_pred" (the
        binarized segmentation prediction after taking the largest connected component). If self.unet.do_ll,
        network_outputs includes "ll_pred" (the network predicted landmark coordinates)
        """
        x = batch["img"]  # Shape: (1, 1, H, W, D) where H, W, D are image width, height, and depth

        # run network
        logits, hmap = monai.inferers.sliding_window_inference(
            x, roi_size=self.sw_roi_rize, sw_batch_size=self.sw_batch_size, predictor=self.unet)

        log_data = {}
        max_data = {}
        network_outputs = {}
        total_loss = 0

        # analyze seg
        if self.unet.do_seg:
            y = batch["seg"]  # Seg target mask; Shape (1, 1, H, W, D)
            y = torch.clip(y, min=0.0, max=1.0)  # this should be unnecessary but is included just in case

            # seg loss
            seg_loss = self.seg_criterion(input=logits, target=y)  # Shape: (1,)
            total_loss += seg_loss
            log_data[f"{prefix}/loss/seg_loss"] = seg_loss

            # binarize prediction and target to get segmentation performance
            # the network outputs logits for each voxel. A logit value of at least 0 corresponds to a probability of at
            # least 0.5 which leads to a voxel classification of True (i.e., voxel is part of segmentation)
            seg_pred = 1.*(logits >= 0)  # Shape: (1, 1, H, W, D)
            seg_targ = y >= 0.5  # Shape: (1, 1, H, W, D)

            if not self.trainer.training:
                # get the largest connected component. this is our finalized segmentation prediction
                # this step is currently disabled during training because it causes a huge slowdown when using multi-GPU
                # training. The slowdown seems to occur when moving data to CPU to run a CPU-only Scipy function.
                seg_pred = utils.get_largest_component(seg_pred.to(torch.float))

            # analyze boundary of segmentation
            #bound_targ = self.boundary_finder.obtain_boundary(y).unsqueeze(1)  # Shape(1, 1, H, W, D)
            #bound_pred = self.boundary_finder.obtain_boundary(seg_pred).unsqueeze(1)  # Shape: (1, 1, H, W, D)

            # get the Dice and IOU for the whole segmentation as well as just the boundary
            preds = [seg_pred.to(torch.bool)]#, bound_pred.to(torch.bool)]
            targs = [seg_targ.to(torch.bool)]#, bound_targ.to(torch.bool)]
            labels = ["seg"]#, "bound"]

            for pred, targ, label in zip(preds, targs, labels):
                # TODO: consider using monai.metrics for Dice and IOU
                dice_key = f"{prefix}/seg/{label}_dice"
                iou_key = f"{prefix}/seg/{label}_iou"
                log_data[dice_key], log_data[iou_key] = utils.get_dice_and_iou(pred, targ)

            # these metrics are expensive, so we do less frequently
            epoch_cond = (self.current_epoch % 5 == 0) or (self.current_epoch == self.trainer.max_epochs-1)
            if epoch_cond:
                hausdorff_key = f"{prefix}/seg/hausdorff"
                assd_key = f"{prefix}/seg/assd"
                log_data[hausdorff_key] = monai.metrics.compute_hausdorff_distance(seg_pred, seg_targ)
                log_data[assd_key] = monai.metrics.compute_average_surface_distance(seg_pred, seg_targ, symmetric=True)

                var_keys = [hausdorff_key, assd_key]
            else:
                var_keys = []

            for key in var_keys:
                max_data[f"{key}_max"] = log_data[key]

            # record the segmentation output
            #network_outputs["seg_pred"] = seg_pred

        # analyze landmark localization
        if self.unet.do_ll:
            z = batch["ll"]  # Heatmap; Shape: (1, 1, H, W, D)

            # get various hmap loss: MSE, WBCE, and whatever the default is
            hmap_loss = self.hmap_criterion(input=hmap, target=z)  # Shape: (1, L, H, W, D). L is number of lanmdarks
            hmap_loss = hmap_loss.mean(dim=(-1, -2, -3)).squeeze()  # Avg over spatial dims. Shape: (L,)

            # logging aggregate losses
            log_data[f"{prefix}/loss/hmap_loss"] = hmap_loss.mean()
            total_loss += self.lmbda * hmap_loss.mean()

            # analyze landmark localization target coordinate
            # pred_coords and targ_coords each consist of a list of 3D coordinates (one per landmark)
            pred_coords = utils.multidim_argmax(hmap)  # Shape: (1, L, 3)

            pred_coords = torch.from_numpy(pred_coords).to(z.device)
            target_coords = utils.multidim_argmax(z)  # Shape: (1, L, 3)
            target_coords = target_coords.cpu()
            ll_euclid_distance = utils.distance(pred_coords, target_coords, pixdim=None).squeeze()  # Shape: (L,)

            # Logging Landmark localization metrics (per landmark and mean)
            log_data[f"{prefix}/ll/dist/mean"] = ll_euclid_distance.mean()
            for idx in range(self.unet.n_landmarks):  # iterate over landmarks
                log_data[f"{prefix}/ll/loss/L{idx}"] = hmap_loss[idx]
                log_data[f"{prefix}/ll/dist/L{idx}"] = ll_euclid_distance[idx]
                max_data[f"{prefix}/ll/dist/L{idx}_max"] = ll_euclid_distance[idx]

            # record the segmentation output
            #network_outputs["ll_pred"] = pred_coords

        on_step = prefix == "test"
        # on_step = False
        self.log_dict(log_data, on_epoch=True, on_step=on_step, batch_size=1, sync_dist=True)
        self.log_dict(max_data, on_epoch=True, on_step=on_step, batch_size=1, sync_dist=True, reduce_fx="max")

        return network_outputs

    def validation_step(self, batch, batch_idx):
        """See _shared_val_test_step for details"""
        assert batch["img"].shape[0] == 1, "Batch size should always be 1 for validation."

        if self.current_epoch >= 5:
            self._shared_val_test_step(batch, prefix="val")

        # log time for model tracking
        #self.log("time_log", float(self.global_step), batch_size=1, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """See _shared_val_test_step for details"""
        assert batch["img"].shape[0] == 1, "Batch size should always be 1 for testing."

        self._shared_val_test_step(batch, prefix="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step

        This function is used when applying the trained network to new data with no label. This function
        runs the U-Net segmentation and landmark localization (depending on do_seg, do_ll) and saves the output in the
        same folder as the input CT scan. Batch size should always be 1 for this.
        """
        assert batch["img"].shape[0] == 1, "Batch size should always be 1 for prediction."

        # get the input image (CT or MRI scan)
        x = batch["img"]  # Shape: (1, 1, H, W, D)

        # extract the filename of the input image from the MetaTensor's meta dictionary
        img_filename = x.meta["filename_or_obj"][0]

        # create filenames
        seg_filename = img_filename.replace(".nii.gz", f"_{self.seg_postfix}.nii.gz")
        ll_filename = img_filename.replace(".nii.gz", f"_{self.ll_postfix}.npy")

        seg_saver = monai.transforms.SaveImaged(
            keys='pred_seg',
            output_dir=os.path.dirname(img_filename),
            output_postfix=self.seg_postfix,
            output_ext='nii.gz',
            resample=False,
            dtype=torch.int8,
            squeeze_end_dims=True,
            separate_folder=False,
            writer="NibabelWriter"
        )
        plt_dir = os.path.join(os.path.dirname(img_filename), "images")
        plt_filename = os.path.join(
            plt_dir,
            str(os.path.basename(img_filename).replace(".nii.gz", f"_flat_visual"))
        )
        if self.unet.do_seg and self.unet.do_ll:
            os.makedirs(plt_dir, exist_ok=True)

        # check if this prediction has already been done
        done = True
        if self.unet.do_seg:
            done = done and (os.path.exists(seg_filename))
        if self.unet.do_ll:
            done = done and (os.path.exists(ll_filename))

        if done and (not self.pred_redo):
            print(f"File: {img_filename} has already been processed! Skipping.")
            return

        # run network
        instance = {}
        try:
            instance = self._predict_step(batch, batch_idx, seg_saver, ll_filename, plt_filename)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(e)
            print("Error! Most likely CUDA ran out of Memory. Retrying with CPU.")
            try:
                # Reset the tracing operation for this transform. CUDA OOM probably left it as False
                spacing_trans = self.trainer.datamodule.eval_pipeline.transforms[2]
                spacing_trans.spacing_transform.sp_resample.tracing = True

                # Change device on post_eval_pipeline
                orig_device = self.trainer.datamodule.post_pipeline.transforms[0].converter.device
                self.trainer.datamodule.post_pipeline.transforms[0].converter.device = torch.device('cpu')

                # Rerun on CPU
                instance = self._predict_step(batch, batch_idx, seg_saver, ll_filename, plt_filename)

                # Restore original device
                self.trainer.datamodule.post_pipeline.transforms[0].converter.device = orig_device

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"Failed to recover from error! Skipping this scan{img_filename}.")
                import traceback as tb
                print(tb.format_exc())
                self.skipped_preds.append(img_filename)

        # finally:  # explicitly free memory from GPU. TODO: Possibly remove or streamline.
        #     del batch
        #     gc.collect()
        #     torch.cuda.empty_cache()

    def _predict_step(self, batch, batch_idx, seg_saver, ll_filename, plt_filename):
        """Main prediction functionality"""
        assert len(batch['img']) == 1, "Prediction only implemented for batch size of 1."

        # get the input image (CT or MRI scan)
        x = batch["img"]  # Shape: (1, 1, H, W, D)
        # extract the filename of the input image from the MetaTensor's meta dictionary
        img_filename = x.meta["filename_or_obj"][0]

        if self.pred_debug:
            # Save the input image before it goes into U-Net. This image has been processed by eval_pipeline
            post_eval_pipe_pred_saver = monai.transforms.SaveImaged(
                keys='img',
                output_dir=os.path.dirname(img_filename),
                output_postfix=self.pep_postfix,
                output_ext='nii.gz',
                resample=False,
                squeeze_end_dims=True,
                separate_folder=False,
                writer="NibabelWriter"
            )
            for instance in monai.data.decollate_batch(batch):
                post_eval_pipe_pred_saver(instance)

        # x must have shape (batch, channel, height, width, depth). In our case the shape is: (1, 1, H, W, D).
        batch['pred_seg'], batch['pred_ll'] = monai.inferers.sliding_window_inference(
            x, roi_size=self.sw_roi_rize, sw_batch_size=self.sw_batch_size, predictor=self.unet,
            overlap=self.sw_overlap, mode=self.sw_mode
        )

        instance = None
        for instance in monai.data.decollate_batch(batch):
            # The U-Net pre-processing involves transformations like changing the voxel spacing and orientation.
            # The post-processing pipeline inverts these transforms so that the segmentation and landmark prediction
            # are in terms of the input's original voxel spacing.
            # This Post-processing pipeline also gets the largest connected component and fills in holes in
            # predicted mask.
            instance = self.trainer.datamodule.post_pipeline(instance)

            if self.unet.do_seg:
                # saves segmentation
                seg_saver(instance)

            if self.unet.do_ll:
                hmap = instance['pred_ll']  # Shape: (L, H, W, D)
                predicted_coords = utils.multidim_argmax(hmap[None]).squeeze()  # Shape: (L, 3)
                predicted_coords = predicted_coords.cpu()
                np.save(ll_filename, predicted_coords)

            if self.unet.do_seg and self.unet.do_ll:
                utils.create_flat_seg_with_landmarks(instance['pred_seg'], predicted_coords,
                                                     filename=plt_filename, flip=True, show=False)
        return instance

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.trainer.max_epochs, power=0.9, last_epoch=-1, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def filter_completed_predictions(self, predict_files):
        # Remove already completed predictions if self.pred_redo = False
        print("Filtering already completed predictions...")
        num_skipped = 0
        if self.pred_redo:
            print("Note: pred_redo flag set to True so any already completed predictions will be reprocessed.")
            return predict_files
        else:
            filtered_rows = []
            for row in predict_files:
                img_fp = row['img']
                seg_fp = img_fp.replace(".nii.gz", f"_{self.seg_postfix}.nii.gz")
                ll_fp = img_fp.replace(".nii.gz", f"_{self.ll_postfix}.npy")
                done = True
                if self.unet.do_seg:
                    done = done and (os.path.exists(seg_fp))
                if self.unet.do_ll:
                    done = done and (os.path.exists(ll_fp))

                if done:
                    if self.pred_verbose:
                        print(f"File: {img_fp} has already been processed! Will not reprocess.")
                    num_skipped += 1
                else:
                    filtered_rows.append(row)
            print(f"In total {num_skipped} input scans will be skipped since they are already processed. To reprocess, "
                  f"rerun this script with '--pred_verbose true' appended to the command.")
            return filtered_rows
