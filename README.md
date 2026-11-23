# EquiCap

When a crystal graph neural network predicts a material property more accurately, is that improvement really due to the crystal structure, or could it come from other parts of the pipeline?

Crystal graph neural networks are widely used to predict material
properties, but it is often unclear whether accuracy gains come from
structural information or from differences in graph construction,
chemical features, training, and evaluation. This matters because
composition alone predicts formation energy well. Equivariant lay-
ers also do not guarantee that a complete predictor is invariant.
We introduce EquiCap, a controlled framework that measures how
much structure improves prediction and audits the symmetry of the
complete pipeline. EquiCap compares distance-only CGCNN-style,
scalar–vector PaiNN-inspired, and composition-only backbones
with attention, capsule-routing, and mean/max readouts. All nine
configurations share the embedding, head, optimizer, split, and
evaluation code. On MP-20 formation energy, the best structural
model reaches a test MAE of 29.2 meV/atom and improves on the
best composition-only model by 4.7 meV/atom (95% CI 4.0 to 5.4).
This headroom is modest, because the readout alone changes the
composition-only error by 7.9 meV/atom. The scalar–vector back-
bone is less accurate than the distance-only backbone with every
readout, by 6.3 to 8.5 meV/atom. Once the audit transforms edges
exactly and the image search is complete, every predictor is invari-
ant to within 3.3 × 10−14 eV/atom in double precision. A stale edge
list or a capped image search produces apparent drifts of up to
2.6 eV/atom and 7.8 × 10−2 eV/atom. Apparent symmetry failures
can therefore come from the evaluation protocol and not from the
model

![EquiCap architecture](figures/F_architecture.pdf)
---

## Install

```bash
git clone https://github.com/OWNER/equicap.git
cd equicap
pip install -e ".[dev]"      # or: pip install -e .
```

PyTorch is not pinned to a build here — install the wheel that matches your CUDA
version first (see pytorch.org), then install this package.

## Verify the install

```bash
python -m equicap --self-test
```

This needs no dataset. It constructs every `backbone × readout` cell on synthetic
crystals and asserts that:

- supplied and rebuilt neighbour graphs give identical predictions;
- gradients are finite for every cell;
- every symmetry in the contract below holds to `1e-8` in float64, with both
  fixed and rebuilt edges;
- a checkpoint round trip reproduces the predictions;
- the two *deliberate* symmetry-breaking controls do break their symmetry —
  without this, a passing audit would not be evidence of anything;
- a rock-salt cell gets 18 neighbours inside 4.5 Å, i.e. whole coordination
  shells rather than an arbitrary top-K truncation.

## Run

```bash
# one cell
equicap --data-root /path/to/MP20 --backbone painn --readout capsule --epochs 200

# the full grid plus controls, three seeds
equicap --data-root /path/to/MP20 --factorial --seeds 0 1 2
```

From Python or a notebook:

```python
from equicap import run, run_factorial, self_test

self_test()
model, res = run(data_root="/path/to/MP20", backbone="painn", readout="capsule")
print(res["test"]["mae"], res["audits"]["symmetry_trained_rebuilt"])

df = run_factorial(data_root="/path/to/MP20", seeds=(0, 1, 2))
```

Every key in `equicap.config.default_config()` is accepted as a keyword to
`run()` and as a `--kebab-case` CLI flag; the parser is generated from that one
dict, so the two can never drift apart.

## Expected data layout

```
data_root/
  train/  train_config.json  train.npz  train.csv
  val/    val_config.json    val.npz    val.csv
  test/   test_config.json   test.npz   test.csv
```

`*.npz` holds a pickled `graph_dict` mapping `material_id` to a structure record
(`node_features`, `cart_coords` or `frac_coords`, `lattice_matrix`, optionally
`space_group`, `edge_index`, `edge_offsets`). `*.csv` holds the targets and
**must** have a `material_id` column — positional alignment of targets to
structures is not safe and is rejected.

## The symmetry contract

`equicap.audit.symmetry_audit` checks the trained predictor against six
transforms. For `chirality=False` and `global_features != "full"`, all six drifts
must be at numerical zero:

| Transform | What it tests |
|---|---|
| rotation | O(3) invariance of the scalar channels |
| reflection | parity evenness (allowed to break only with `chirality=True`) |
| translation | no dependence on the origin of the cell |
| permutation | no dependence on atom ordering |
| periodic image | no dependence on which image an atom is written in |
| supercell | intensivity under exact replication |

Two audit modes, and the difference between them is the point:

- `edge_mode="fixed"` transforms the **edge list along with the structure**,
  remapping indices and periodic offsets so each edge still points at the same
  physical neighbour. Drift here is the network and its numerics, nothing else.
- `edge_mode="rebuilt"` lets the neighbour search run again on the transformed
  structure, so drift now also includes neighbour selection and image
  enumeration.

Run both in float32 and float64 (`dtype=torch.float64`) to separate arithmetic
invariance from rounding. `per_structure=True` returns per-crystal drift arrays,
which is what you need to split the reflection test by centrosymmetric versus
Sohncke space groups (`equicap.spacegroups`).

`equicap.audit.audit_edges` is the separate, earlier check: it recomputes edge
lengths from coordinates, lattice and stored offsets. Mass beyond the cutoff
means the mined graph is wrong and `edge_source="stored"` must not be used.

## Things worth knowing before you report numbers

- **The MAD ratio, not R².** `compute_metrics` reports `mad_over_mae`: the
  mean-absolute-deviation baseline divided by the model's MAE. A value near 1
  means the model has learned nothing useful whatever the R² says.
- **Polymorph multiplicity.** `CrystalPropertyDataset.composition_stats()`
  reports structures per reduced composition. If that ratio is near 1, a
  composition-only baseline can in principle match any structural model, and the
  `composition` backbone is the control that shows it.
- **The radius cap.** When an atom has more than `max_deg` neighbours inside the
  cutoff, the graph is no longer a radius graph. The affected atom count is
  tracked per split (`n_overflow_atoms`), warned about, and stored in the
  checkpoint under `audits["overflow_atoms"]`. Report it.
- **`global_features="full"` is an ablation, not a default.** It is deliberately
  not supercell-invariant and the self-test asserts that it breaks.
- **`probe="legacy"`** reproduces `<v_i,v_j>/(|v_i||v_j|+1e-6)`, whose value at a
  symmetry-cancelled site is the direction of a float32 rounding residual. It
  exists so the paper can quantify what the smooth probe fixes.
- **Routing normalisation.** At the first routing iteration `atoms` and `mass`
  differ only by a factor `1/M`; the scale-matched control is `mass` with
  `temperature = n_caps` (the `massT` variant). Later iterations differ
  genuinely. See `MotifCapsules` for the derivation.
- **`neighbor_mode="knn"`** keeps the `max_deg` nearest pairs *that also fall
  inside the cutoff*, since the RBF envelope is defined on that cutoff. It is the
  graph-construction ablation, not a second default.

## Repository layout

| Path | Contents |
|---|---|
| `equicap/utils.py` | seeding, one-shot warnings, environment probes |
| `equicap/geometry.py` | norms, lattice descriptors, periodic neighbour list |
| `equicap/spacegroups.py` | centrosymmetric / Sohncke tables |
| `equicap/chemistry.py` | element tables, physical descriptors |
| `equicap/layers.py` | RBF, `EquivariantConv`, `CGConv`, capsule blocks |
| `equicap/readouts.py` | attention / routing-weighted / mean-max pooling |
| `equicap/model.py` | `CrystalPropertyModel` |
| `equicap/transforms.py` | symmetry transforms with exact edge remapping |
| `equicap/audit.py` | `symmetry_audit`, `audit_edges` |
| `equicap/data.py` | dataset, collate, shared element vocabulary |
| `equicap/metrics.py` | MAE / RMSE / R² / r / MAD ratio |
| `equicap/training.py` | EMA, LR schedule, train and validate loops |
| `equicap/checkpoint.py` | save / load / rebuild-from-config |
| `equicap/inference.py` | prediction tables, ad-hoc structure scoring |
| `equicap/config.py` | defaults, layout check, run tags |
| `equicap/experiment.py` | one cell (`main`) and the grid (`run_factorial`) |
| `equicap/cli.py` | argparse front end |
| `equicap/selftest.py` | synthetic end-to-end check |
| `tests/` | pytest suite (`-m "not slow"` skips the full self-test) |
| `docs/ARCHITECTURE.md` | dependency graph and design notes |

## Outputs

Each cell writes, under `<ckpt_dir>` (default `<data_root>/propertycheckpoints`):

- `<tag>.pt` — weights, full model config, target standardisation, training
  history, both metric dicts, and every audit result;
- `<tag>_metrics.csv` — per-epoch training curve;
- `<tag>_test_predictions.csv` — **per-structure** predictions, errors, atom
  counts and space groups, which is what makes a paired error analysis between
  two cells reproducible;
- `<tag>_distance_hist.npy` — edge-length histogram from the graph audit.

`run_factorial` additionally writes `factorial_results.csv` after *every* cell,
so an interrupted sweep still leaves usable output.

## Tests

```bash
pytest -q -m "not slow"     # fast unit tests
pytest -q                   # including the full self-test
ruff check equicap tests
```

## License

MIT — see [LICENSE](LICENSE).
