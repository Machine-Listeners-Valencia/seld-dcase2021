import os
from metrics import SELD_evaluation_metrics
import cls_feature_class
import parameter
import numpy as np


class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'], 'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass(params)
        
        # collect reference files
        self._ref_files = os.listdir(self._desc_dir)
        self._nb_ref_files = len(self._ref_files)

        self._ref_labels = {}
        for ref_file in self._ref_files:
            # Load reference description file
            gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, ref_file))
            if not self._use_polar_format:
                gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
            self._ref_labels[ref_file] = self._feat_cls.segment_labels(gt_dict, self._feat_cls.get_nb_frames())

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._feat_cls.get_nb_frames())

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file])

        # Overall SED and DOA scores
        er, f, de, de_f = eval.compute_seld_scores()
        seld_scr = SELD_evaluation_metrics.early_stopping_metric([er, f], [de, de_f])

        return er, f, de, de_f, seld_scr

    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = self._feat_cls.segment_labels(pred_dict, self._feat_cls.get_nb_frames())

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file])

                # Overall SED and DOA scores
                er, f, de, de_f = eval.compute_seld_scores()
                seld_scr = SELD_evaluation_metrics.early_stopping_metric([er, f], [de, de_f])

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(er, 100*f))
                print('DOA metrics: DOA error: {:0.1f}, F-score:{:0.1f}'.format(de, 100*de_f))

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    ref_files_folder = 'results/metadata_dev'
    pred_output_format_files = 'results/4_foa_dev'

    score_obj = ComputeSELDResults(parameter.get_params(), ref_files_folder=ref_files_folder)
    er, f, de, de_f, seld_scr = score_obj.get_SELD_Results(pred_output_format_files)
    print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
    print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(er, 100*f))
    print('DOA metrics: DOA error: {:0.1f}, F-score:{:0.1f}'.format(de, 100*de_f))

    score_obj.get_consolidated_SELD_results(pred_output_format_files)
