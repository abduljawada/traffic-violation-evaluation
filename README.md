# Traffic Violation Detection Evaluation Suite

A comprehensive Python evaluation framework for evaluating traffic violation detection submissions against ground truth data.

## Overview

This repository contains tools and scripts for:
- **Evaluating** traffic violation detection model submissions
- **Generating** video clips and frames from source videos
- **Processing** annotation data and ground truth information
- **Analyzing** results using multiple evaluation metrics

The evaluation system compares submissions against ground truth using:
- **Categorical Fields**: Macro F1-score
- **Description Fields**: Average of normalized CIDEr and BERTScore
- **Final Score**: Mean of all individual field scores

## Features

**Comprehensive Evaluation Metrics**
- Categorical field evaluation using F1-scores
- Text similarity evaluation using CIDEr and BERTScore
- Flexible time tolerance for event timing

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/abduljawada/traffic-violation-evaluation.git
cd traffic-violation-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Evaluation

Evaluate a submission JSON file against ground truth:

```bash
python evaluate.py submission.json --gt groundtruth.json
```

## Submission Format

Submissions should be a flat JSON array where each object describes one exported clip. `clip_name` is the clip identifier used to align submission rows to ground-truth rows when both files provide unique clip names. `clip_name` is used for matching only and is not scored as a categorical field.

```json
[
  {
    "clip_name": "0_0_001.mp4",
    "date": "2018-07-17",
    "time": "06:01:20",
    "violation_type": "wrong_way",
    "violator_type": "car",
    "color": "dark",
    "entering_direction": "upper left",
    "entering_lane": "1",
    "exiting_direction": "bottom left",
    "exiting_lane": "3",
    "intersection_type": "T-intersection",
    "weather": "clear",
    "light": "daylight",
    "description": "Dark car traveling against the designated direction of traffic."
  }
]
```

The evaluator also accepts the previous grouped format, `[{ "video_id": "...", "violations": [...] }]`, for backwards compatibility. In both formats, `video_id`, `clip_name`, `clip_export_name`, `start_time`, and `end_time` are ignored during scoring.

### Supported Violation Types
- `wrong_way`
- `uturn` (U-turn)
- `jaywalking`
- `red_light`
- `lane_use_control`
- `lane_discipline`
- `no_violation`

### Supported Values

| Field | Options |
|-------|---------|
| `violator_type` | car, motorcycle, pedestrian, bus, truck, na |
| `color` | dark, light, red, green, yellow, blue, brown, purple, pink, orange, gray, mixed, na |
| `entering_direction`, `exiting_direction` | upper left, upper right, bottom left, bottom right, left, right, up, down, na |
| `entering_lane`, `exiting_lane` | 1, 2, 3, 4, na |
| `intersection_type` | T-intersection, four-way intersection |
| `weather` | clear, rainy, cloudy |
| `light` | daylight, night |

## Ground Truth Format

Ground truth data is stored in `groundtruth.json` with the same flat structure as submissions. If both ground truth and submission files contain unique `clip_name` values, rows are aligned by `clip_name`; otherwise, rows are compared in file order.

## Evaluation Metrics

### Categorical Fields
Macro-averaged F1-score, calculated across all unique values for balance and fairness.

### Description Fields
Combined score from:
- **CIDEr** (Consensus-based Image Description Evaluation): Measures semantic similarity through n-gram matching
- **BERTScore**: Measures semantic similarity using contextual embeddings

Final description score = (normalized_cider + bertscore) / 2

### Overall Score
Mean of all individual field scores, providing a holistic performance metric.

## Project Structure

```
.
├── groundtruth.json           # Ground truth data
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .github/                   # GitHub configuration
    ├── CONTRIBUTING.md       # Contributing guidelines
    └── ISSUE_TEMPLATE/       # Issue templates
```

## Evaluation Results

Evaluation results are saved as JSON files with detailed scores:

```json
{
  "submission_file": "submission.json",
  "ground_truth_file": "groundtruth.json",
  "total_score": 0.85,
  "per_field_scores": {
    "violation_type": 0.92,
    "violator_type": 0.88,
    "color": 0.82,
    ...
  },
  "time_violations": 5,
  "summary": "Overall performance evaluation summary"
}
```

## Time Tolerance

The evaluation uses a default time tolerance of **7 seconds** when matching violations across submissions and ground truth. This accounts for variations in timestamp precision.

## Examples

### Evaluate Multiple Submissions
```bash
for file in submission_*.json; do
  python evaluate.py "$file" --gt groundtruth.json
done
```

### Generate Report
```bash
python evaluate.py submission.json --gt groundtruth.json > results.txt
```

## Troubleshooting

### BERTScore Model Download
First run may download the BERT model (~1GB). This is normal and only happens once.

### Memory Issues
For large datasets, consider:
- Evaluating submissions in batches
- Increasing system memory
- Reducing batch size in processing scripts

### JSON Validation
Ensure submission files are valid JSON:
```bash
python -m json.tool submission.json > /dev/null && echo "Valid JSON"
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on how to:
- Report bugs
- Suggest features
- Submit pull requests

## Citation

If you use this evaluation suite in your research, please cite:

```bibtex
TBD
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Authors

TBD

## Acknowledgments

- Evaluation metrics based on established computer vision benchmarks
- BERT-Score from [Zhang et al.](https://github.com/Tiiiger/bert_score)
- CIDEr implementation based on [Vedantam et al.](https://github.com/salaniz/pycocoevalcap)

## Support

For issues, questions, or suggestions, please:
- Open an issue on GitHub
- Check existing documentation
- Review example submissions

## Changelog

### Version 1.0.0 (April 2026)
- Initial release
- Comprehensive evaluation metrics
- Video processing utilities
- Batch processing capabilities

---

**For more information**, please refer to the inline documentation in the source files or open an issue on GitHub.
