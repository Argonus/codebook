## TensorFlow Loss Monitor

This monitor is used to monitor the loss of the model.
After each epoch, the monitor will print the loss of the model and save it to a file.

### Loss Metrics

| Column | Description | Usage | Example | Good Range |
|--------|-------------|--------|---------|------------|
| epoch | Current training epoch | Track progress over time | 1 | Any positive integer |
| timestamp | ISO format timestamp | Track when metrics were collected | 2025-02-23T20:55:35 | Valid ISO timestamp |
| class_name | Name of the predicted class | Identify class-specific performance | 'pneumonia' | Any valid class name |
| high_conf | Ratio of high confidence predictions (>0.9) | Monitor strong predictions | 0.45 | > 0.3 |
| med_conf | Ratio of medium confidence (0.6-0.9) | Track reasonable certainty | 0.35 | 0.3 - 0.5 |
| uncertain | Ratio of uncertain predictions (0.4-0.6) | Identify confusion cases | 0.15 | < 0.2 |
| low_conf | Ratio of low confidence (<0.4) | Monitor weak predictions | 0.05 | < 0.1 |
| true_positives | Count of correct positive predictions | Track true detections | 156 | Class dependent |
| false_positives | Count of incorrect positive predictions | Monitor false alarms | 12 | < 20% of true_positives |
| loss_contribution | Class contribution to total loss | Track problematic classes | 0.234 | < 0.5 |
| avg_confidence_correct | Mean confidence for correct predictions | Monitor true prediction strength | 0.89 | > 0.8 |
| avg_confidence_incorrect | Mean confidence for wrong predictions | Check overconfident mistakes | 0.34 | < 0.4 |

#### High Confidence Ratio

- Description: Proportion of predictions with >90% confidence
- Formula: count(pred > 0.9) / total_predictions
- Good Range: > 0.3
- Warning Signs:
  - Too low: Model is uncertain
  - Too high: Possible overconfidence
- Actions if Bad:
  - Adjust focal loss gamma
  - Review data quality
  - Check class balance

#### Medium Confidence Ratio

- Description: Proportion of predictions with 60-90% confidence
- Formula: count(0.6 < pred ≤ 0.9) / total_predictions
- Good Range: 0.3 - 0.5
- Warning Signs:
  - Too low: Model might be overconfident
  - Too high: Model lacks certainty
- Actions if Bad:
  - Review decision boundary
  - Check feature quality
  - Consider data augmentation

#### Uncertain Predictions

- Description: Proportion of borderline predictions (40-60%)
- Formula: count(0.4 < pred ≤ 0.6) / total_predictions
- Good Range: < 0.2
- Warning Signs:
  - High ratio indicates confusion
  - Growing ratio suggests degrading performance
- Actions if Bad:
  - Add more training data
  - Review similar classes
  - Consider feature engineering

#### Low Confidence Ratio

- Description: Proportion of low confidence predictions (<40%)
- Formula: count(pred ≤ 0.4) / total_predictions
- Good Range: < 0.1
- Warning Signs:
  - High ratio indicates poor learning
  - Increasing ratio suggests problems
- Actions if Bad:
  - Check data quality
  - Review preprocessing
  - Consider model capacity

#### True/False Positives
- Description: Correct/incorrect positive predictions
- Formula:
  - TP: count((pred > 0.5) & (true == 1))
  - FP: count((pred > 0.5) & (true == 0))
- Good Range: FP < 20% of TP
- Warning Signs:
  - High FP rate
  - Decreasing TP rate
- Actions if Bad:
  - Adjust decision threshold
  - Review class balance
  - Consider focal loss alpha

#### Loss Contribution
- Description: Class-specific contribution to total loss
- Formula: Depends on loss function (BCE or Focal)
- Good Range: < 0.5
- Warning Signs:
  - High contribution from one class
  - Unstable contributions
- Actions if Bad:
  - Review class weights
  - Check data balance
  - Adjust focal loss parameters

#### Average Confidence Metrics

- Description: Mean confidence for correct/incorrect predictions
- Formula:
  - Correct: mean(pred where pred matches true)
  - Incorrect: mean(pred where pred doesn't match true)
- Good Range:
  - Correct: > 0.8
  - Incorrect: < 0.4
- Warning Signs:
  - Low confidence in correct predictions
  - High confidence in mistakes
- Actions if Bad:
  - Adjust focal loss gamma
  - Review similar classes
  - Consider model calibration
