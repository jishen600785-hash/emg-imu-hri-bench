# Result inventory and interpretation

## Classification

| Result | Accuracy | Macro-F1 | Source file |
|---|---:|---:|---|
| UCI selected fusion model | 0.822744 | 0.822581 | `results/public_datasets/uci/UCI_STRICT_DEEP_FEATURE_SUMMARY.csv` |
| CapgMyo selected spatial-temporal model | 0.952160 | 0.952278 | `results/public_datasets/capgmyo/STRICT_ADVANCED_MODEL_SUMMARY.csv` |
| Limb personalized aggregate | 0.905556 | 0.905003 | `results/public_datasets/limb/PERSONALIZED_SUMMARY.csv` |
| Bingbin V3 recording aggregate | 0.928571 | 0.927211 | `results/public_datasets/bingbin/BINGBIN_V3_SUMMARY.csv` |
| SJ improved raw dynamic test | 0.810714 | 0.802615 | `results/sj_training/SJ_MODEL_SUMMARY.csv` |
| SJ causal seven-window dynamic test | 0.921429 | 0.920782 | `results/sj_training/SJ_MODEL_SUMMARY.csv` |

These values are not pooled because the datasets, class counts, subjects, and evaluation protocols differ. The Bingbin V3 result remains exploratory because earlier results on the reused test split were already known.

## Robot task

| Metric | Limb | SJ V3 | SJ V4 |
|---|---:|---:|---:|
| Trials | 42 | 42 | 42 |
| Hits | 35 | 34 | 42 |
| Misses | 4 | 2 | 0 |
| Timeouts | 3 | 6 | 0 |
| Success rate | 0.8333 | 0.8095 | 1.0000 |
| Classification accuracy | 0.9265 | 0.8405 | 0.9020 |
| Mean simulated movement time (s) | 17.343 | 18.977 | 11.942 |
| Mean throughput (bit/s) | 0.1623 | 0.1634 | 0.3708 |
| Mean inference latency (ms) | 7.111 | 3.710 | 4.769 |
| Mean control latency (ms) | 26.839 | 25.423 | 26.993 |
| Direction switches | 502 | 1,020 | 109 |

Compact machine-readable files are stored as `results/robot/<branch>/metrics.json`. The V4 change is a controller-policy improvement: width-aware stopping, axis hysteresis, and distance-dependent speed scaling reduce oscillation and timeouts. It does not retrain the classifier.

Wall-clock execution can be slower than simulated movement time when Gazebo runs below real-time speed. Reported Fitts-style movement time uses the ROS simulation clock.
