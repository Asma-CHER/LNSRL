# KES 2025 results

`Results_KES2025.xlsx` is preserved byte-for-byte from the research workspace.
It contains 41 benchmark rows followed by the aggregate row.

The main `KES2025` sheet contains these method groups:

- `RAND`: classical random-destroy LNS;
- `SISR`: SISR destroy baseline;
- `DPR`: Dynamic Partial Removal;
- `DR-ALNS`: comparison method reported in the paper;
- `LNS-RL_V0`: the no-route-information ablation, labelled
  `LNS-RL-NoRoute` in the compact sheets;
- `LNS-RL`: the paper method associated with `models/lns_rl_kes.pth`.

For each method, the full sheet reports best distance, average distance, best
CPU time, and average CPU time. The compact `Best`, `Average`, and `Average
CPU` sheets are exported to correspondingly named CSV files for use without
spreadsheet software.

The OPT column is retained exactly as stored in the workbook, including its
decimal-comma text representation.
