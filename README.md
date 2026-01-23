# Day-to-Day Route–Departure Choice Dynamics with Two-Stage Bayesian Perception Updating

This repository contains the research code and materials for a study on day-to-day (DTD) traffic dynamics with simultaneous route and departure-time (SRD) choice under imperfect and lagged travel information.

## Research Overview

We study how travelers form and update their perceived travel times using a **two-stage Bayesian perception updating scheme**, which integrates:
- **Pre-trip information** (broadcast before the trip)
- **Post-trip experience** (actual travel time after the trip)

Given perceived costs (including schedule delay penalties), travelers choose route–departure alternatives according to a **random-utility Logit model**, while within-day congestion is represented by a **static Bureau of Public Roads (BPR)** performance function.

## Key Features

### Information Structures
We compare three information provision scenarios:
- **S0 (No information)**: Experience-only learning
- **S1 (Historical broadcast)**: System-wide historical information
- **S2 (Information sharing)**: Peer-to-peer experience sharing with reliability depending on previous-day choice proportion

### Experimental Settings
- **Networks**: Toy network and Sioux Falls case study
- **Disruption scenarios**: Capacity reduction during specified days
- **Metrics**: Travel time, relative gap, switching behaviors, departure-time patterns

### Main Findings
Numerical experiments highlight an **efficiency–stability trade-off**:
- Sufficiently reliable broadcast information can mitigate peak disruption congestion
- However, it may induce stronger day-to-day re-allocation
- Share-dependent information can delay return-to-baseline due to herding effects

## Paper Status

📝 **Manuscript in preparation**

The full paper, code implementation, and experimental data will be made available upon publication.

## Contact

For questions or collaboration inquiries, please contact:
- Gen Li: gen_li1993@163.com

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026dtd,
  title={Day-to-Day Route--Departure Choice Dynamics with Two-Stage Bayesian Perception Updating under Alternative Information Structures},
  author={Li, Gen and Xu, Pengcheng and Lan, Jieyuan},
  journal={Under review},
  year={2026}
}
```

## License

This repository is currently private. Code and data will be released under an open-source license upon paper acceptance.
