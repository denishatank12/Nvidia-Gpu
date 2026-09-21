# HW2.5 — GPU performance measurements

`HW2_5_GPU_Assignment.ipynb` is the submission notebook. It is configured to refuse execution on hardware other than the required RTX 4090 or RTX 5090, and it writes UUID-labelled measurements, figures, logs, and reports.

The directory `evidence/rtx5060ti/` contains the supplied executed 5060 Ti artifacts for provenance and review. They are deliberately kept separate from the submission outputs because an RTX 5060 Ti does not satisfy the lab's RTX 4090/5090 requirement.

## Run on the assigned workstation

```bash
python -m pip install -r requirements.txt
jupyter notebook HW2_5_GPU_Assignment.ipynb
```

Run every cell on the reserved RTX 4090 or RTX 5090 workstation. The sustained-load cell must run for the full 20 minutes. Before submission, add the real reservation record and GPU-hour total to `RESERVATION_GPU_HOURS.md`, inspect the thermal evidence, and review the generated `METRICS.md` and `RUN_LOG.txt`.

See `SUBMISSION_AUDIT.md` for the requirement-by-requirement status and the exact evidence still needed.

The required Git tag is `hw2-5`.
