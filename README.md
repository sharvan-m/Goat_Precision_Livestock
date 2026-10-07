# GoatVision Clean Repository

Source:
D:\weight_detect

Clean runtime repository:
D:\goat_repo

This repository contains the application source and runtime AI artifacts only.

Excluded intentionally:
- Python virtual environments
- node_modules
- .next
- Python caches
- training datasets
- temporary uploads
- old backup files
- local .env secrets
- historical/unused training scripts

IMPORTANT:
The model files included here are the model artifacts currently present in
D:\weight_detect. Their actual use by each backend service should be audited
before AWS deployment.

No file named:
weight_img_ds1_seed42.pt

was found in the current project inventory, so it was NOT copied.

Next step:
1. Start backend.
2. Start frontend.
3. Test each AI endpoint.
4. Trace every endpoint to its exact model.
5. Remove unused model artifacts after that audit.
