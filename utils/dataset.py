# coding: utf-8

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os, copy
import pandas as pd
import numpy as np
import time
#from torch.utils.data import Dataset


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.dataset_path = os.path.abspath(config['data_path'])
        self.preprocessed_dataset_path = os.path.abspath(config['preprocessed_data'])
        self.preprocessed_loaded = False        # if preprocessed data loaded?
        self.logger = getLogger()
        self.dataset_name = config['dataset']

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.ts_id = self.config['TIME_FIELD']

        if df is not None:
            self.df = df
            return
        self.ui_core_splitting_str = self._k_core_and_splitting()
        self.processed_data_name = '{}_{}_processed.inter'.format(self.dataset_name, self.ui_core_splitting_str)
        # load from preprocessed path?
        if self.config['load_preprocessed'] and self._load_preprocessed_dataset():
            self.preprocessed_loaded = True
            self.logger.info('\nData loaded from preprocessed dir: ' + self.preprocessed_dataset_path + '\n')
            return
        # load dataframe
        self._from_scratch()
        # pre-processing
        self._data_processing()

    def _k_core_and_splitting(self):
        user_min_n = 1
        item_min_n = 1
        if self.config['min_user_inter_num'] is not None:
            user_min_n = max(self.config['min_user_inter_num'], 1)
        if self.config['min_item_inter_num'] is not None:
            item_min_n = max(self.config['min_item_inter_num'], 1)
        # splitting
        ratios = self.config['split_ratio']
        tot_ratio = sum(ratios)
        # remove 0.0 in ratios
        ratios = [i for i in ratios if i > .0]
        ratios = [str(int(_ * 10 / tot_ratio)) for _ in ratios]
        s = ''.join(ratios)
        return 'u{}i{}_s'.format(user_min_n, item_min_n) + s

    def _load_preprocessed_dataset(self):
        file_path = os.path.join(self.preprocessed_dataset_path, self.processed_data_name)
        if not os.path.isfile(file_path):
            return False
        # load
        self.df = self._load_df_from_file(file_path, self.config['load_cols']+[self.config['preprocessed_data_splitting']])
        return True

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.info('Loading {} from scratch'.format(self.__class__))
        # get path
        file_path = os.path.join(self.dataset_path, '{}.inter'.format(self.dataset_name))
        if not os.path.isfile(file_path):
            raise ValueError('File {} not exist'.format(file_path))
        self.df = self._load_df_from_file(file_path, self.config['load_cols'])

    def _load_df_from_file(self, file_path, load_columns):
        # read header(user_id:token   item_id:token   rating:float    timestamp:float) for ml-10k
        cnt = 0
        with open(file_path, 'r') as f:
            head = f.readline()[:-1]
            field_separator = self.config['field_separator']
            # only use [user_id, item_id, timestamp]
            for field_type in head.split(field_separator):
                if field_type in load_columns:
                    cnt += 1
            # all cols exist
            if cnt != len(load_columns):
                raise ValueError('File {} lost some required columns.'.format(file_path))

        df = pd.read_csv(file_path, sep=self.config['field_separator'], usecols=load_columns)
        return df

    def _data_processing(self):
        """Data preprocessing, including:
        - K-core data filtering
        - Remap ID
        """
        # drop N/A value
        self.df.dropna(inplace=True)
        # remove duplicate rows
        self.df.drop_duplicates(inplace=True)
        # perform k-core
        self._filter_by_k_core(self.df)
        # remap ID
        self._reset_index(self.df)

    def _filter_by_k_core(self, df):
        """Filter by number of interaction.

        Upper/Lower bounds can be set, only users/items between upper/lower bounds can be remained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Lower bound is also called k-core filtering, which means this method will filter loops
            until all the users and items has at least k interactions.
        """
        while True:
            ban_users = self._get_illegal_ids_by_inter_num(df, field=self.uid_field,
                                                           max_num=self.config['max_user_inter_num'],
                                                           min_num=self.config['min_user_inter_num'])
            ban_items = self._get_illegal_ids_by_inter_num(df, field=self.iid_field,
                                                           max_num=self.config['max_item_inter_num'],
                                                           min_num=self.config['min_item_inter_num'])

            if len(ban_users) == 0 and len(ban_items) == 0:
                return

            dropped_inter = pd.Series(False, index=df.index)
            if self.uid_field:
                dropped_inter |= df[self.uid_field].isin(ban_users)
            if self.iid_field:
                dropped_inter |= df[self.iid_field].isin(ban_items)
            # self.logger.info('[{}] dropped interactions'.format(len(dropped_inter)))
            df.drop(df.index[dropped_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(self, df, field, max_num=None, min_num=None):
        """Given inter feat, return illegal ids, whose inter num out of [min_num, max_num]

        Args:
            field (str): field name of user_id or item_id.
            feat (pandas.DataFrame): interaction feature.
            max_num (int, optional): max number of interaction. Defaults to ``None``.
            min_num (int, optional): min number of interaction. Defaults to ``None``.

        Returns:
            set: illegal ids, whose inter num out of [min_num, max_num]
        """
        self.logger.debug('\n get_illegal_ids_by_inter_num:\n\t field=[{}], max_num=[{}], min_num=[{}]'.format(
            field, max_num, min_num
        ))

        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = df[field].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(ids), field))
        return ids

    def _reset_index(self, df):
        if df.empty:
            raise ValueError('Some feat is empty, please check the filtering settings.')
        df.reset_index(drop=True, inplace=True)

    def split(self, ratios):
        """Split interaction records by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            group_by (str, optional): Field name that interaction records should grouped by after splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been splitted.

        Note:
            Other than the first one, each part is rounded down.
        """
        if self.preprocessed_loaded:
            dfs = []
            splitting_label = self.config['preprocessed_data_splitting']
            # splitting into training/validation/test
            for i in range(3):
                temp_df = self.df[self.df[splitting_label] == i].copy()
                temp_df.drop(splitting_label, inplace=True, axis=1)
                dfs.append(temp_df)
            # wrap as RecDataset
            full_ds = [self.copy(_) for _ in dfs]
            return full_ds

        tot_ratio = sum(ratios)
        # remove 0.0 in ratios
        ratios = [i for i in ratios if i > .0]
        ratios = [_ / tot_ratio for _ in ratios]

        # get split global time
        split_ratios = np.cumsum(ratios)[:-1]
        split_timestamps = list(np.quantile(self.df[self.ts_id], split_ratios))

        # get df training dataset unique users/items
        df_train = self.df.loc[self.df[self.ts_id] < split_timestamps[0]]
        self.logger.info('==Splitting: 1. Reindexing and filtering out new users/items not in train dataset...')

        uni_users = pd.unique(df_train[self.uid_field])
        uni_items = pd.unique(df_train[self.iid_field])
        # re_index users & items
        u_id_map = {k: i for i, k in enumerate(uni_users)}
        i_id_map = {k: i for i, k in enumerate(uni_items)}
        self.df[self.uid_field] = self.df[self.uid_field].map(u_id_map)
        self.df[self.iid_field] = self.df[self.iid_field].map(i_id_map)
        # filter out Nan line
        self.df.dropna(inplace=True)
        # as int
        self.df = self.df.astype(int)

        # split df based on global time
        self.logger.info('==Splitting: 2. Train/Valid/Test.')
        dfs = []
        start = 0
        for i in split_timestamps:
            dfs.append(self.df.loc[(start <= self.df[self.ts_id]) & (self.df[self.ts_id] < i)].copy())
            start = i
        # last
        dfs.append(self.df.loc[start <= self.df[self.ts_id]].copy())

        # save to disk
        self.logger.info('==Splitting: 3. Dumping...')
        self._save_dfs_to_disk(u_id_map, i_id_map, dfs)
        # self._drop_cols(dfs+[self.df], [self.ts_id])

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def _save_dfs_to_disk(self, u_map, i_map, dfs):
        if self.config['load_preprocessed'] and not self.preprocessed_loaded:
            dir_name = self.preprocessed_dataset_path
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # save id mapping
            u_df = pd.DataFrame(list(u_map.items()), columns=[self.uid_field, 'new_id'])
            i_df = pd.DataFrame(list(i_map.items()), columns=[self.iid_field, 'new_id'])
            u_df.to_csv(os.path.join(self.preprocessed_dataset_path,
                                     '{}_u_{}_mapping.csv'.format(self.dataset_name, self.ui_core_splitting_str)),
                        sep=self.config['field_separator'], index=False)
            i_df.to_csv(os.path.join(self.preprocessed_dataset_path,
                                     '{}_i_{}_mapping.csv'.format(self.dataset_name, self.ui_core_splitting_str)),
                        sep=self.config['field_separator'], index=False)
            # 0-training/1-validation/2-test
            for i, temp_df in enumerate(dfs):
                temp_df[self.config['preprocessed_data_splitting']] = i
            temp_df = pd.concat(dfs)
            temp_df.to_csv(os.path.join(self.preprocessed_dataset_path, self.processed_data_name),
                           sep=self.config['field_separator'], index=False)
            self.logger.info('\nData saved to preprocessed dir: \n' + self.preprocessed_dataset_path)

    # def _drop_cols(self, dfs, col_names):
    #     for _df in dfs:
    #         _df.drop(col_names, inplace=True, axis = 1)

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)
        return nxt

    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.

        Args:
            field (str): field name to get token number.

        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.config['load_cols']:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        uni_len = len(pd.unique(self.df[field]))
        return uni_len

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def sort_by_chronological(self):
        self.df.sort_values(by=[self.ts_id], inplace=True, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        if self.uid_field:
            self.user_num = len(uni_u)
            self.avg_actions_of_users = self.inter_num/self.user_num
            info.extend(['The number of users: {}'.format(self.user_num),
                         'Average actions of users: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            self.item_num = len(uni_i)
            self.avg_actions_of_items = self.inter_num/self.item_num
            info.extend(['The number of items: {}'.format(self.item_num),
                         'Average actions of items: {}'.format(self.avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / self.user_num / self.item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
