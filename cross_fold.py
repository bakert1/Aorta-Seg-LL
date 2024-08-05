import os
from typing import List, Optional, Tuple

import numpy as np


class CrossFoldUtil(object):
    """Class for organizing cross fold validation.
    Cross fold validation is implemented as a class to facilitate use with PyTorch Lightning CLI.

    Can be used to create data CSVs that facilitate cross fold validation.
        Example: Split data into 5 CSVs for 5-fold cross validation. Each fold can then be told which 4 data CSVs
        to use for training and which CSV to use for validation.

    Also supports the creation of a hold-out set that is not part of the cross folds. A hold out set is useful when
    using cross fold validation to tune hyperparameters.
    """
    def __init__(
            self,
            root_dir: Optional[str] = None,
            data_size: int = -1,
            hold_out_size: int = 0,
            num_folds: int = 5,
            seed: int = 42,
            fold_idx: Optional[int] = None
    ):
        """

        Args:
            root_dir:
                root directory for the cross validation data CSVs. Defaults to os.getcwd() if not given.
            data_size:
                size of the dataset
            hold_out_size:
                size of the hold out set
            num_folds:
                number of validation folds
            seed:
                random seed used to create cross fold validation set
            fold_idx:
                Current index of cross fold validation fold. Ex: For 5-fold cross validation, fold_idx can be {0, 1, 2,
                 3, 4}. This parameter is optional. It can be specified either when creating this object or it can be
                specified later when invoking functions.
        """
        assert fold_idx is None or (0 <= fold_idx < num_folds),\
            f"fold_idx:{fold_idx} should be between 0 and number of folds: {num_folds}"

        self.root_dir = root_dir or os.getcwd()
        self.data_size = data_size
        self.hold_out_size = hold_out_size
        self.num_folds = num_folds
        self.seed = seed
        self.fold_idx = fold_idx

    def get_fold_filename(
            self,
            full_path: bool,
            fold_idx: Optional[int] = None,
            verbose: int = True
    ) -> str:
        """Gets the data CSV filename for fold=fold_idx.

        Args:
            full_path:
                If False, return just the data CSV's filename (e.g., d481_h0_n5_s42_i4.csv). If True, return the
                full filepath (e.g., "self.data_root/d481_h0_n5_s42_i4.csv").
            fold_idx:
                The fold index of which to return the data CSV's filename for. This parameter must be given if
                self.fold_idx is None. If both self.fold_idx and fold_idx are specified, then fold_idx takes
                precedence.
            verbose:
                Whether to print out a warning message if both self.fold_idx and fold_idx are both specified.
        Returns:
            The filename of the specified datafold.
        """

        self._check_fold_idx(fold_idx, verbose)
        if fold_idx is None:
            fold_idx = self.fold_idx

        filename = f"d{self.data_size}_h{self.hold_out_size}_n{self.num_folds}_s{self.seed}_i{fold_idx}.csv"
        if full_path:
            filename = os.path.join(self.root_dir, filename)
        return filename

    def get_holdout_filename(
            self,
            full_path: bool
    ) -> str:
        """Gets the data CSV filename for the holdout set.

        Args:
            full_path:
                If False, return just the data hold set CSV filename (e.g., d481_n5_s42_i4.csv). If True, return the
                full filepath (e.g., "self.data_root/d481_n5_s42_i4.csv").
        Returns:
            The filename of the specified datafold.
        """

        filename = f"d{self.data_size}_h{self.hold_out_size}_n{self.num_folds}_s{self.seed}_holdout.csv"
        if full_path:
            filename = os.path.join(self.root_dir, filename)

        return filename

    def create_cross_fold_split(
            self,
            df
    ) -> None:
        """Creates a set of fold CSVs based on self.data_size, self.hold_out_size self.num_folds, and self.seed.
        The input dataframe contains the master list of all data and should have length == self.data_size.

        Args:
            df:
                Pandas Dataframe of all data. All dataframe columns will be written to the cross fold CSVs. For
                example, the df columns can be the CT filename and its corresponding segmentation filename.
        """
        if self.data_size == -1:
            self.data_size = len(df)

        assert len(df) == self.data_size,\
            f"Error: the length of the dataframe: {len(df)} should match this object's data_size: {self.data_size}"

        rand_perm = np.random.RandomState(self.seed).permutation(self.data_size)

        # create the training/validation folds
        effective_size = self.data_size - self.hold_out_size
        min_fold_size = effective_size // self.num_folds
        folds_with_extra = effective_size % self.num_folds

        start_idx = 0
        for idx in range(self.num_folds):
            if idx < folds_with_extra:
                end_idx = start_idx + min_fold_size + 1
            else:
                end_idx = start_idx + min_fold_size

            fold_indices = rand_perm[start_idx:end_idx]
            fold_df = df.iloc[fold_indices]
            start_idx = end_idx

            filepath = self.get_fold_filename(fold_idx=idx, full_path=True)
            fold_df.to_csv(filepath)

        # create the hold test fold
        if self.hold_out_size > 0:
            fold_indices = rand_perm[start_idx:]
            fold_df = df.iloc[fold_indices]
            filepath = self.get_holdout_filename(full_path=True)
            fold_df.to_csv(filepath)

    def _check_fold_idx(
            self,
            fold_idx,
            verbose
    ) -> None:
        """Checks whether self.fold_idx and fold_idx are both specified."""
        if (fold_idx is not None) and (self.fold_idx is not None) and verbose:
            print(f"Warning: Both fold_idx:{fold_idx} and self.fold_idx:{self.fold_idx} specified. The former will take"
                  f"precedent.")

    def get_train_val_test_lists(
            self,
            fold_idx: Optional[int] = None,
            verbose: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """Given a fold idx, returns which (self.num_folds-1) data CSV files are used for training and which data CSV
        is used for validation.

        Args:
            fold_idx:
                the cross fold index
            verbose:
                whether to print a warning message if both self.fold_idx and fold_idx are specified

        Returns:
            training_csvs:
                which (self.num_folds - 1) data CSVs are to be used for training
            val_csvs:
                which data CSV are to be used for validation or testing
        """
        self._check_fold_idx(fold_idx, verbose)

        if fold_idx is None:
            fold_idx = self.fold_idx

        # the train CSVs are all fold CSVs besides the one corresponding to the current fold index
        train_csvs = [self.get_fold_filename(full_path=True, fold_idx=i, verbose=False)
                      for i in range(self.num_folds) if i != fold_idx]

        # the validation CSV is fold CSV corresponding to the current fold index
        val_csvs = [self.get_fold_filename(full_path=True, fold_idx=fold_idx, verbose=False)]

        # the test CSV is the holdout set
        if self.hold_out_size > 0:
            test_csvs = [self.get_holdout_filename(full_path=True)]
        else:
            test_csvs = []

        # some sanity checks
        assert len(train_csvs) == self.num_folds-1, f"{len(train_csvs)}"
        assert len(val_csvs) == 1, f"{len(val_csvs)}"
        assert len(test_csvs) == 1, f"{len(val_csvs)}"
        for csv in train_csvs:
            assert os.path.exists(csv)
        assert os.path.exists(val_csvs[0])
        assert os.path.exists(test_csvs[0])

        return train_csvs, val_csvs, test_csvs
