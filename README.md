# Defense prep

Figures and plotting code for the Stitch evaluation section of Rubens Lacouture's defense talk.

## DLRM plot sequence

`plots/dlrm` contains separate five-frame builds for 16 GB and 100 GB embedding tables. Each frame is available as a slide-ready PNG and a vector PDF.

1. Empty cost and throughput axes
2. FleetRec-style reference point
3. Pareto envelope for the single-decomposition CPU, GPU, and accelerator family
4. Full modeled Stitch search space
5. Overall Pareto frontier

The orange point is a modeled FleetRec-style proxy. It is not a reproduction of the hardware used in the FleetRec paper. The fixed-family curve keeps the execution template constant while varying the CPU, GPU, accelerator, node count, interconnect, and CPU-to-accelerator table split.

The CSV files contain every evaluated configuration. `sequence_metadata.json` records the selected anchor and the two Pareto frontiers.

## Regenerate the figures

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/dlrm_plot_sequence.py
```

The generator reads `scripts/dlrm_pareto.py` and replaces the files under `plots/dlrm`.
