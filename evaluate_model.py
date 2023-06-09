#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py labels outputs scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
# model, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import os, os.path, sys, numpy as np
from helper_code import *

# Evaluate the models.
def evaluate_model(label_folder, output_folder):
    # Load the labels.
    patient_ids = find_data_folders(label_folder)
    num_patients = len(patient_ids)

    hospitals = list()
    label_outcomes = list()
    label_cpcs = list()

    for i in range(num_patients):
        patient_data_file = os.path.join(label_folder, patient_ids[i], patient_ids[i] + '.txt')
        patient_data = load_text_file(patient_data_file)

        hospital = get_hospital(patient_data)
        label_outcome = get_outcome(patient_data)
        label_cpc = get_cpc(patient_data)

        hospitals.append(hospital)
        label_outcomes.append(label_outcome)
        label_cpcs.append(label_cpc)

    # Load the model outputs.
    output_outcomes = list()
    output_outcome_probabilities = list()
    output_cpcs = list()

    for i in range(num_patients):
        output_file = os.path.join(output_folder, patient_ids[i], patient_ids[i] + '.txt')
        output_data = load_text_file(output_file)

        output_outcome = get_outcome(output_data)
        output_outcome_probability = get_outcome_probability(output_data)
        output_cpc = get_cpc(output_data)

        output_outcomes.append(output_outcome)
        output_outcome_probabilities.append(output_outcome_probability)
        output_cpcs.append(output_cpc)

    # Evaluate the models.
    challenge_score = compute_challenge_score(label_outcomes, output_outcome_probabilities, hospitals)
    auroc_outcomes, auprc_outcomes = compute_auc(label_outcomes, output_outcome_probabilities)
    accuracy_outcomes, _, _ = compute_accuracy(label_outcomes, output_outcomes)
    f_measure_outcomes, _, _ = compute_f_measure(label_outcomes, output_outcomes)
    mse_cpcs = compute_mse(label_cpcs, output_cpcs)
    mae_cpcs = compute_mae(label_cpcs, output_cpcs)

    # Return the results.
    return challenge_score, auroc_outcomes, auprc_outcomes, accuracy_outcomes, f_measure_outcomes, mse_cpcs, mae_cpcs

# Compute the Challenge score.
def compute_challenge_score(labels, outputs, hospitals):
    # Check the data.
    assert len(labels) == len(outputs)

    # Convert the data to NumPy arrays for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Identify the unique hospitals.
    unique_hospitals = sorted(set(hospitals))
    num_hospitals = len(unique_hospitals)

    # Initialize a confusion matrix for each hospital.
    tps = np.zeros(num_hospitals)
    fps = np.zeros(num_hospitals)
    fns = np.zeros(num_hospitals)
    tns = np.zeros(num_hospitals)

    # Compute the confusion matrix at each output threshold separately for each hospital.
    for i, hospital in enumerate(unique_hospitals):
        idx = [j for j, x in enumerate(hospitals) if x == hospital]
        current_labels = labels[idx]
        current_outputs = outputs[idx]
        num_instances = len(current_labels)

        # Collect the unique output values as the thresholds for the positive and negative classes.
        thresholds = np.unique(current_outputs)
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        idx = np.argsort(current_outputs)[::-1]

        # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)

        tp[0] = 0
        fp[0] = 0
        fn[0] = np.sum(current_labels == 1)
        tn[0] = np.sum(current_labels == 0)

        # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
        k = 0
        for l in range(1, num_thresholds):
            tp[l] = tp[l-1]
            fp[l] = fp[l-1]
            fn[l] = fn[l-1]
            tn[l] = tn[l-1]

            while k < num_instances and current_outputs[idx[k]] >= thresholds[l]:
                if current_labels[idx[k]] == 1:
                    tp[l] += 1
                    fn[l] -= 1
                else:
                    fp[l] += 1
                    tn[l] -= 1
                k += 1

            # Compute the FPRs.
            fpr = np.zeros(num_thresholds)
            for l in range(num_thresholds):
                if tp[l] + fn[l] > 0:
                    fpr[l] = float(fp[l]) / float(tp[l] + fn[l])
                else:
                    fpr[l] = float('nan')

            # Find the threshold such that FPR <= 0.05.
            max_fpr = 0.05
            if np.any(fpr <= max_fpr):
                l = max(l for l, x in enumerate(fpr) if x <= max_fpr)
                tps[i] = tp[l]
                fps[i] = fp[l]
                fns[i] = fn[l]
                tns[i] = tn[l]
            else:
                tps[i] = tp[0]
                fps[i] = fp[0]
                fns[i] = fn[0]
                tns[i] = tn[0]

    # Compute the TPR at FPR <= 0.05 for each hospital.
    tp = np.sum(tps)
    fp = np.sum(fps)
    fn = np.sum(fns)
    tn = np.sum(tns)

    if tp + fn > 0:
        max_tpr = tp / (tp + fn)
    else:
        max_tpr = float('nan')

    return max_tpr

# Compute area under the receiver operating characteristic curve (AUROC) and area under the precision recall curve (AUPRC).
def compute_auc(labels, outputs):
    assert len(labels) == len(outputs)
    num_instances = len(labels)

    # Convert the data to NumPy arrays for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j-1]
        fp[j] = fp[j-1]
        fn[j] = fn[j-1]
        tn[j] = tn[j-1]

        while i < num_instances and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]] == 1:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs, TNRs, and PPVs at each threshold.
    tpr = np.zeros(num_thresholds)
    tnr = np.zeros(num_thresholds)
    ppv = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = tp[j] / (tp[j] + fn[j])
        else:
            tpr[j] = float('nan')
        if fp[j] + tn[j] > 0:
            tnr[j] = tn[j] / (fp[j] + tn[j])
        else:
            tnr[j] = float('nan')
        if tp[j] + fp[j] > 0:
            ppv[j] = tp[j] / (tp[j] + fp[j])
        else:
            ppv[j] = float('nan')

    # Compute AUROC as the area under a piecewise linear function with TPR/sensitivity (x-axis) and TNR/specificity (y-axis) and
    # AUPRC as the area under a piecewise constant with TPR/recall (x-axis) and PPV/precision (y-axis).
    auroc = 0.0
    auprc = 0.0
    for j in range(num_thresholds-1):
        auroc += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
        auprc += (tpr[j+1] - tpr[j]) * ppv[j+1]

    return auroc, auprc

# Construct the one-hot encoding of data for the given classes.
def compute_one_hot_encoding(data, classes):
    num_instances = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_instances, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for j, y in enumerate(classes):
            if (x == y) or (is_nan(x) and is_nan(y)):
                one_hot_encoding[i, j] = 1

    return one_hot_encoding

# Compute the binary confusion matrix, where the columns are the expert labels and the rows are the classifier labels for the given
# classes.
def compute_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_instances):
        for i in range(num_classes):
            for j in range(num_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A

# Construct the binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier
# for the given classes.
def compute_one_vs_rest_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_instances):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0: # TN
                A[j, 1, 1] += 1

    return A

# Compute accuracy.
def compute_accuracy(labels, outputs):
    # Compute the confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_confusion_matrix(labels, outputs, classes)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float('nan')

    # Compute per-class accuracy.
    num_classes = len(classes)
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(labels[:, i]) > 0:
            per_class_accuracy[i] = A[i, i] / np.sum(A[:, i])
        else:
            per_class_accuracy[i] = float('nan')

    return accuracy, per_class_accuracy, classes

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    # Compute confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes

# Compute mean-squared error.
def compute_mse(labels, outputs):
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mse = np.mean((labels - outputs)**2)

    return mse

# Compute mean-absolute error.
def compute_mae(labels, outputs):
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mae = np.mean(np.abs(labels - outputs))

    return mae

if __name__ == '__main__':
    # Compute the scores for the model outputs.
    scores = evaluate_model(sys.argv[1], sys.argv[2])

    # Unpack the scores.
    challenge_score, auroc_outcomes, auprc_outcomes, accuracy_outcomes, f_measure_outcomes, mse_cpcs, mae_cpcs = scores

    # Construct a string with scores.
    output_string = \
        'Challenge Score: {:.3f}\n'.format(challenge_score) + \
        'Outcome AUROC: {:.3f}\n'.format(auroc_outcomes) + \
        'Outcome AUPRC: {:.3f}\n'.format(auprc_outcomes) + \
        'Outcome Accuracy: {:.3f}\n'.format(accuracy_outcomes) + \
        'Outcome F-measure: {:.3f}\n'.format(f_measure_outcomes) + \
        'CPC MSE: {:.3f}\n'.format(mse_cpcs) + \
        'CPC MAE: {:.3f}\n'.format(mae_cpcs)

    # Output the scores to screen and/or a file.
    if len(sys.argv) == 3:
        print(output_string)
    elif len(sys.argv) == 4:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
