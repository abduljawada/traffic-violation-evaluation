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

For detailed results with individual field scores:
```bash
python evaluate.py submission.json --gt groundtruth.json -v
```

## Submission Format

Submissions should be in JSON format with the following structure:

```json
[
  {
    "video_id": "video_001",
    "violations": [
      {
        "violation_type": "wrong_way",
        "violator_type": "car",
        "date": "2024-01-15",
        "time": "14:30:00",
        "color": "red",
        "entering_direction": "left",
        "entering_lane": "1",
        "exiting_direction": "right",
        "exiting_lane": "2",
        "intersection_type": "four-way intersection",
        "weather": "clear",
        "light": "daylight",
        "description": "Red car turning left from wrong lane at intersection"
      }
    ]
  }
]
```

### Supported Violation Types
- `wrong_way`
- `uturn` (U-turn)
- `crossing`
- `red_light`
- `wrong_lane`
- `illegal_lane_switching`

### Supported Values

| Field | Options |
|-------|---------|
| `violator_type` | car, motorcycle, pedestrian, bus, truck |
| `color` | dark, light, red, green, yellow, blue, brown, purple, pink, orange, gray, mixed |
| `direction` | upper_left, upper_right, bottom_left, bottom_right, left, right, up, down |
| `lane` | 1, 2, 3, 4 |
| `intersection_type` | T-intersection, four-way intersection |
| `weather` | clear, rainy, cloudy |
| `light` | daylight, night, dawn, dusk |

## Ground Truth Format

Ground truth data is stored in `groundtruth.json` with the same structure as submissions.

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
