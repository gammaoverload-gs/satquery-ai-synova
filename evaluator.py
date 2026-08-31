import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class RemoteSensingBenchmarkEvaluator:
    """
    Evaluates SatQuery AI on prescribed remote-sensing benchmarks:
    - RSVQA (Single-image VQA accuracy)
    - VRSBench (Captioning & Visual Grounding mIoU/BLEU)
    - CDVQA (Bi-temporal Change VQA & F1)
    """
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def calculate_bbox_iou(self, box_pred: list, box_gt: list) -> float:
        """Calculates Intersection over Union for [ymin, xmin, ymax, xmax]"""
        ymin_p, xmin_p, ymax_p, xmax_p = box_pred
        ymin_g, xmin_g, ymax_g, xmax_g = box_gt

        inter_ymin = max(ymin_p, ymin_g)
        inter_xmin = max(xmin_p, xmin_g)
        inter_ymax = min(ymax_p, ymax_g)
        inter_xmax = min(xmax_p, xmax_g)

        inter_area = max(0, inter_ymax - inter_ymin) * max(0, inter_xmax - inter_xmin)
        pred_area = max(0, ymax_p - ymin_p) * max(0, xmax_p - xmin_p)
        gt_area = max(0, ymax_g - ymin_g) * max(0, xmax_g - xmin_g)
        union_area = pred_area + gt_area - inter_area

        return inter_area / (union_area + 1e-8)

    def evaluate_text_response(self, prediction: str, ground_truth: str) -> dict:
        """Computes BLEU-4 and ROUGE-L for VQA and captioning outputs"""
        ref_tokens = [ground_truth.lower().split()]
        pred_tokens = prediction.lower().split()

        bleu4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)
        rouge_scores = self.rouge.score(ground_truth, prediction)

        # Exact match approximation for short VQA answers
        exact_match = 1.0 if ground_truth.strip().lower() in prediction.strip().lower() else 0.0

        return {
            "bleu_4": round(bleu4, 4),
            "rouge_l": round(rouge_scores['rougeL'].fmeasure, 4),
            "exact_match": exact_match
        }

    def run_benchmark_suite(self, predictions: list, ground_truths: list, benchmark_type: str = "RSVQA") -> dict:
        """
        Runs comprehensive evaluation over a batch of predictions.
        """
        bleu_list = []
        rouge_list = []
        exact_matches = []

        for pred, gt in zip(predictions, ground_truths):
            metrics = self.evaluate_text_response(pred, gt)
            bleu_list.append(metrics["bleu_4"])
            rouge_list.append(metrics["rouge_l"])
            exact_matches.append(metrics["exact_match"])

        return {
            "Benchmark": benchmark_type,
            "Total Samples": len(predictions),
            "Mean BLEU-4": round(float(np.mean(bleu_list)), 4),
            "Mean ROUGE-L": round(float(np.mean(rouge_list)), 4),
            "Accuracy / Exact Match": round(float(np.mean(exact_matches)) * 100.0, 2)
        }

# Pre-defined test sample runner
if __name__ == "__main__":
    evaluator = RemoteSensingBenchmarkEvaluator()
    sample_preds = ["Built-up area has expanded with newly constructed residential blocks.", "Water body is localized in the center."]
    sample_gts = ["Built-up area increased with new buildings.", "Water body located in central region."]
    
    results = evaluator.run_benchmark_suite(sample_preds, sample_gts, "CDVQA_Change_Benchmark")
    print(json.dumps(results, indent=4))