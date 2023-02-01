class MetricUtils:


    @staticmethod
    def get_numbers(ground_truths, predictions):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(ground_truths)):
            if ground_truths[i] == 1 and predictions[i] == 1:
                tp += 1
            elif ground_truths[i] == 1 and predictions[i] == 0:
                fn += 1
            elif ground_truths[i] == 0 and predictions[i] == 1:
                fp += 1
            elif ground_truths[i] == 0 and predictions[i] == 0:
                tn += 1
            return tp, tn, fp, fn

    @staticmethod
    def get_metrics(tp, tn, fp, fn):
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = None
        recall = None
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        if (tp + fn) != 0:
            recall = tp / (tp + fn)

        return accuracy, precision, recall

    @staticmethod
    def f1_score(accuracy, precision, recall):
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

