import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import monai
import monai.data as mon_data
import monai.transforms as mon_trans
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

import constants


class SegLLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            image_type: str = "Old",
            do_seg: bool = True,
            do_ll: bool = True,
            data_root: Optional[str] = None,
            cache_dir: Optional[str] = None,
            data_csvs: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None,
            batch_size: int = 8,
            num_random_crops: int = 4,
            crop_size: Tuple[int, int, int] = constants.DEFAULT_CROP_SIZE,
            crop_threshold: Optional[float] = None,
            crop_weight_bckg: float = 1.,
            crop_weight_seg: float = 1.,
            crop_weight_ll: Optional[List[float]] = None,
            load_only: bool = False,
            device: Optional[str] = None,
            low_memory_predict: bool = False
    ):
        """Data module for segmentation (seg) and/or landmark localization (ll) 3D image datasets.

        Defaults for all parameters are given to facilitate use of Lightning Commandline interface (CLI)

        See latest documentation on PyTorch Lightning for more info on DataModules:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html

        Args:
            image_type:
                The type of medical image. One of "CTA", "MRA", "Old". Currently, this setting only affects voxel
                intensity normalization. See _update_pipelines() for more information.
            do_seg:
                Whether to use and process segmentation data.
            do_ll:
                Whether to use and process ll data. Set to False if no landmark labels are available.
            data_root:
                root directory to prepend to any filepath in the data CSVs.
            cache_dir:
                Directory of where to save cached data. See also monai.data.CacheNTransDataset
            data_csvs:
                dictionary containing up to four keys: ['train', 'val', 'test', 'pred']. Each key maps to a list of CSV
                filepaths. For example, data_csvs['train'] should be a list of CSV filepaths to use for training while
                data_csvs['val'] should be a list of CSV filepaths to use for validation. You can also pass a single CSV
                filepath for each key instead of a list of filepaths. Each CSV should have up to three columns:
                [img, seg, ll]. img is always required and should point to the input medical scan NIFTI. seg is also
                always required and should point to a segmentation label NIFTI; ll points to a heatmap label NIFTI.
            batch_size:
                Batch size used for training dataloader.
            num_random_crops:
                Number of random crops used in the training pipeline.
            crop_size:
                3-tuple defining the size of random crops used in training pipeline: (height, width, depth).
            crop_threshold:
                Threshold for determining voxels to serve as random crop centers. See also
                 monai.transforms.RandomCropByLabelClasses.
            crop_weight_bckg:
                Weight that controls probability of centering random crop on background.
            crop_weight_seg:
                Weight that controls probability of centering random crop on segmentation.
            crop_weight_ll:
                Weight that controls probability of centering random crop on ll.
            device:
                Torch device (e.g., 'cpu', 'cuda:0', 'cuda:1', etc.) to put the data on.
        """
        assert image_type in ["CTA", "MRA", "Old"], f"Invalid image type given! {image_type}."
        super(SegLLDataModule, self).__init__()
        self.num_landmarks = constants.NUM_LANDMARKS if do_ll else 0

        self.image_type = image_type
        self.do_seg = do_seg
        self.do_ll = do_ll
        self.data_root = data_root or ""
        self.cache_dir = cache_dir
        self.data_csvs = data_csvs or {}
        self.batch_size = batch_size
        self.num_random_crops = num_random_crops
        self.crop_size = crop_size
        self.load_only = load_only
        self.device = device
        self.low_memory_predict = low_memory_predict

        self.crop_probs = self._get_crop_probs(crop_weight_bckg, crop_weight_seg, crop_weight_ll)

        if crop_threshold is None:
            self.crop_image_key = None
            self.crop_threshold = 0.0  # this value does not matter; setting to 0.0 avoids type warnings from MONAI.
        else:
            self.crop_image_key = 'img'
            self.crop_threshold = crop_threshold

        # the following are defined in setup() which must always be called before using DataModule
        self.keys, self.modes, self.dtypes = None, None, None
        self.train_pipeline = None
        self.eval_pipeline = None
        self.post_pipeline = None
        self.train, self.val, self.test, self.predict = (None,)*4
        self.num_cache = None

        # stores all provided init arguments into a dictionary called self.hparams.
        self.save_hyperparameters()

    def prepare_data(self):
        """
        This method is not needed for our pipeline. This method may be useful when using a distributed training
        strategy. More info: https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(
            self,
            stage: Optional[str] = None
    ) -> None:
        """ Sets up the data pipeline.

        Args:
            stage:
                Generally one of ["fit", "validate", "test", "predict"] corresponding to how the data will be used.
        """

        self.keys, self.modes, self.dtypes = self._get_keys_modes_and_dtypes(stage)
        self.update_pipelines()

        if stage == 'fit':
            train_files = self._extract_files_from_csvs(self.data_csvs.get("train", []))
            val_files = self._extract_files_from_csvs(self.data_csvs.get("val", []))

            self.train = monai.data.CacheNTransDataset(
                data=train_files, transform=self.train_pipeline, cache_dir=self.cache_dir, cache_n_trans=self.num_cache
            )
            self.val = monai.data.CacheNTransDataset(
                data=val_files, transform=self.eval_pipeline, cache_dir=self.cache_dir, cache_n_trans=self.num_cache
            )

        elif stage == 'validate':
            val_files = self._extract_files_from_csvs(self.data_csvs.get("val", []))
            self.val = monai.data.Dataset(data=val_files, transform=self.eval_pipeline)
        elif stage == "test":
            test_files = self._extract_files_from_csvs(self.data_csvs.get("test", []))
            self.test = monai.data.Dataset(data=test_files, transform=self.eval_pipeline)

        elif stage == "predict":
            predict_files = self._extract_files_from_csvs(self.data_csvs.get("pred", []))
            # remove already completed files if model is set up to do so.
            predict_files = self.trainer.model.filter_completed_predictions(predict_files)
            print("Loading U-Net pipeline...")
            self.predict = monai.data.Dataset(data=predict_files, transform=self.eval_pipeline)

    def train_dataloader(self):
        return mon_data.ThreadDataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=False,
                                         num_workers=0, buffer_size=1)

    def test_dataloader(self):
        return mon_data.DataLoader(self.test, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    def val_dataloader(self):
        return mon_data.ThreadDataLoader(self.val, batch_size=1, shuffle=False, pin_memory=False,
                                         num_workers=0, buffer_size=1)

    def predict_dataloader(self):
        return mon_data.DataLoader(self.predict, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    def _get_crop_probs(self, crop_weight_bckg, crop_weight_seg, crop_weight_ll):
        crop_weight_ll = crop_weight_ll or [1. for _ in range(self.num_landmarks)]
        assert len(crop_weight_ll) == self.num_landmarks,\
            f"{len(crop_weight_ll)} and {self.num_landmarks} should match."

        crop_weight_seg = 0 if not self.do_seg else crop_weight_seg
        crop_weight_ll = [0 for _ in crop_weight_ll] if not self.do_ll else crop_weight_ll

        crop_weights = np.array([crop_weight_bckg, crop_weight_seg, *crop_weight_ll])
        return crop_weights / crop_weights.sum()

    def _get_keys_modes_and_dtypes(self, stage):
        keys = ["img"]  # always use medical image
        modes = ['bilinear']
        dtypes = [torch.float]
        if self.do_seg and (stage != "predict"):  # prediction implies no labels
            keys.append("seg")
            modes.append("nearest")
            dtypes.append(torch.bool)
        if self.do_ll and (stage != "predict"):
            keys.append("ll")
            modes.append('bilinear')
            dtypes.append(torch.half)
        return keys, modes, dtypes

    def _get_pipelines(self) -> Tuple[mon_trans.Transform, mon_trans.Transform, mon_trans.Transform]:
        """Defines the training, evaluation, and post-processing pipelines.

        Read more about the transforms at MONAI's website: https://docs.monai.io/en/stable/transforms.html

        Returns:
            the training, evaluation, and post-processing pipelines.
        """

        ####################################################
        # Transforms used for both training and evaluation #
        ####################################################
        load = mon_trans.LoadImaged(
            keys=self.keys,
            reader="NibabelReader",
            ensure_channel_first=True,
            image_only=True
        )
        orient = mon_trans.Orientationd(
            keys=self.keys,
            axcodes="RAI"
        )
        ct_clip = mon_trans.ScaleIntensityRanged(
            keys="img",
            a_min=-1000, a_max=1000,
            b_min=-1000, b_max=1000,
            clip=True
        )
        ct_clip_old = mon_trans.ScaleIntensityRanged(
            keys="img",
            a_min=-1000, a_max=1000,
            b_min=0, b_max=1,
            clip=True
        )
        zscore_normalize = mon_trans.NormalizeIntensityd(
            keys="img",
        )
        ct_z_normalize = mon_trans.NormalizeIntensityd(
            keys="img",
            subtrahend=constants.CT_AORTA_MEAN,
            divisor=constants.CT_AORTA_STD
        )
        spacing = mon_trans.Spacingd(
            keys=self.keys,
            pixdim=np.array([1., 1., 1.]),
            mode=self.modes
        )
        crop_foreground = mon_trans.CropForegroundd(
            keys=self.keys,
            source_key="img",
            allow_smaller=True
        )
        ensure_type = mon_trans.EnsureTyped(
            keys=self.keys,
            data_type="tensor",
            dtype=self.dtypes
        )
        ensure_type_gpu = mon_trans.EnsureTyped(
            keys=self.keys,
            data_type="tensor",
            dtype=self.dtypes,
            device=self.device
        )
        ensure_type_gpu_eval = mon_trans.EnsureTyped(
            keys=self.keys,
            data_type="tensor",
            device=self.device
        )

        ######################################
        # Transforms used only for training  #
        ######################################
        pad = mon_trans.SpatialPadd(
            keys=self.keys,
            spatial_size=self.crop_size
        )
        seg_crop_indices = mon_trans.ClassesToIndicesd(
            keys='seg',
            num_classes=2,
            image_key=self.crop_image_key,
            image_threshold=self.crop_threshold,
            max_samples_per_class=10000,
            allow_missing_keys=True,  # mostly for eval pipeline
        )
        ll_crop_indices = mon_trans.ClassesToIndicesd(
            keys='ll',
            image_key=self.crop_image_key,
            image_threshold=self.crop_threshold,
            max_samples_per_class=1000,
            allow_missing_keys=True,  # mostly for eval pipeline
        )
        rand_crop_seg = mon_trans.RandCropByLabelClassesd(
            keys=self.keys,
            label_key='seg',
            spatial_size=self.crop_size,
            ratios=self.crop_probs[0:2],
            num_classes=2,
            num_samples=self.num_random_crops,
            indices_key='seg_cls_indices'
        )
        rand_crop_ll = mon_trans.RandCropByLabelClassesd(
            keys=self.keys,
            label_key='ll',
            spatial_size=self.crop_size,
            ratios=self.crop_probs[2:],
            num_samples=self.num_random_crops,
            indices_key='ll_cls_indices'
        )
        rand_crop = mon_trans.OneOf(
            transforms=[rand_crop_seg, rand_crop_ll],
            weights=[sum(self.crop_probs[0:2]), sum(self.crop_probs[2:])]
        )
        delete_indices = mon_trans.DeleteItemsd(
            keys=['seg_cls_indices', 'll_cls_indices'],
            use_re=False
        )
        rand_brightness = mon_trans.RandScaleIntensityFixedMeand(
            keys="img",
            prob=0.25,
            factors=0.3
        )
        rand_contrast_pos = mon_trans.RandAdjustContrastd(
            keys="img",
            prob=0.25,
            gamma=1.5,
            invert_image=False,
            retain_stats=True
        )
        rand_contrast_inv = mon_trans.RandAdjustContrastd(
            keys="img",
            prob=0.25,
            gamma=1.5,
            invert_image=True,
            retain_stats=True
        )
        rand_contrast = mon_trans.OneOf(
            transforms=[rand_contrast_pos, rand_contrast_inv],
            weights=[0.5, 0.5]
        )
        rand_rotate = mon_trans.RandRotated(
            keys=self.keys,
            prob=0.75,
            range_x=0.5, range_y=0.5, range_z=0.5
        )

        # Determine how to normalize voxel intensities
        if self.image_type == "CTA":
            normalize = monai.transforms.Compose([ct_clip, ct_z_normalize], lazy=True)
            self.num_cache = 10
        elif self.image_type == "MRA":
            normalize = zscore_normalize
            self.num_cache = 9
        elif self.image_type == "Old":
            normalize = ct_clip_old
            self.num_cache = 9
        else:
            raise NotImplementedError(f"Invalid image type given: {self.image_type}.")

        #############################################
        # Compose evaluation and training pipelines #
        #############################################
        train_pipeline = mon_trans.Compose(
            [
                load, orient, spacing, crop_foreground, normalize, ensure_type,  # standardize the images
                pad, seg_crop_indices, ll_crop_indices,  # prepare for random crop
                rand_crop, delete_indices, ensure_type_gpu,  # perform random crop and move data to GPU
                rand_brightness, rand_contrast, rand_rotate  # other data augmentation
            ],
            lazy=True
        ).flatten()

        # The ensure_type twice in a row is not a mistake.
        # The plain ensure_type ensures that the CacheDataset stores tensors in certain datatypes.
        # For example, the segmentations are saved as bool array. (this saves a lot of memory and some time).
        # The ensure_type_gpu moves the data to GPU. Cache-ing ensure_type_gpu breaks the code
        eval_pipeline = mon_trans.Compose(
            transforms=[
                load, orient, spacing, crop_foreground, normalize, ensure_type,  # standardize the images
                pad, seg_crop_indices, ll_crop_indices,  # only included for cache reasons
                ensure_type_gpu_eval  # move data to GPU
            ],
            lazy=True
        ).flatten()

        if self.load_only:
            train_pipeline, eval_pipeline = load, load

        ####################################
        # Compose post processing pipeline #
        ####################################
        # Post processing pipeline depends on evaluation pipeline, and so needs to define after eval_pipeline exists.
        # Helpful tutorial for this part:
        #   - https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py
        pred_keys = []
        pred_device = "cpu" if self.low_memory_predict else self.device
        if self.do_seg:
            pred_keys.append('pred_seg')
        if self.do_ll:
            pred_keys.append('pred_ll')

        pred_ensure_type = mon_trans.EnsureTyped(
            keys=pred_keys,
            data_type="tensor",
            device=pred_device
        )

        sigmoid = mon_trans.Activationsd(
            keys=pred_keys,
            sigmoid=True,
        )

        pred_to_cpu = mon_trans.EnsureTyped(
            keys=pred_keys,
            data_type="tensor",
            device=torch.device('cpu')
        )

        # The evaluation pipeline transforms the input image including changing its voxel spacing and shape. The
        # following transform applies the inversion of the evaluation pipeline so that the final prediction corresponds
        # to the input image's original shape and voxel spacing.
        undo_eval_pipeline = mon_trans.Invertd(
            keys=pred_keys,
            transform=eval_pipeline,
            orig_keys='img',
            nearest_interp=False,  # recommended by MONAI. Use AsDiscreted to binarize segmentation instead
            to_tensor=True
        ) if not self.load_only else None

        #####################################
        # Segmentation only post processing #
        #####################################
        binarize = mon_trans.AsDiscreted(
            keys='pred_seg',
            threshold=0.5,
        )
        largest_connected = mon_trans.KeepLargestConnectedComponentd(
            keys='pred_seg',
        )
        fill_holes = mon_trans.FillHolesd(
            keys='pred_seg',
        )
        ################################
        # Heatmap only post processing #
        ################################
        post_pipeline = mon_trans.Compose([
            pred_ensure_type, sigmoid,  # prepare to invert evaluation pipeline
            undo_eval_pipeline,  # invert evaluation transforms
            binarize, largest_connected, fill_holes,  # post process segmentation results
            pred_to_cpu,  # move data to cpu
        ])

        return train_pipeline, eval_pipeline, post_pipeline

    def _extract_files_from_csvs(self, files):
        """Extract img, segmentation, and heatmap file names from `files`.

        Files can be:
        - a single CSV filepath (type: str)
        - a list of CSV filepaths (type: List[str])
        - a directory containing medical scans for prediction (type: str). Only use this option when you are running a
        prediction workflow as it offers no avenue for specifying segmentations or landmark files which are needed for
        training, validating, and testing.
        """
        if type(files) is str:
            if os.path.isdir(files):
                print(f"{files} should be a directory containing medical image files! Only use this option when running "
                      f"predictions. All NIFTI files in this directory will be passed through the U-Net, except those "
                      f"containing the string 'VOI'.")
                # Gather all files
                root = files  # files is a root directory
                possible_files = os.listdir(root)
                all_files = []
                for file in possible_files:
                    if file.endswith(".nii.gz") and "VOI" not in file:
                        all_files.append({"img": os.path.join(root, file)})
            else:
                files = [files]

        if type(files) is list:
            all_files = []
            for csv in files:
                df = pd.read_csv(csv)
                for key in self.keys:
                    # CSV file paths should be stored as Windows filepaths. However, the following code *may* not break
                    # if the CSV file paths are actually stored as Linux ones. The following line converts the filepath
                    # based on which platform (Windows or Linux) is currently being used.
                    df[key] = df.apply(
                        lambda row: os.path.join(self.data_root, str(pathlib.PurePath(pathlib.PureWindowsPath(row[key])))),
                        axis=1
                    )
                all_files.extend(df.to_dict("records"))

        return all_files

    def update_pipelines(self):
        self.train_pipeline, self.eval_pipeline, self.post_pipeline = self._get_pipelines()
