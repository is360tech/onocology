# Governance and licence

## MIAS licence (Licence.txt in the database; summary, not legal advice)
Use is granted for research purposes only; the database may not be rented, leased or sold; the database must be referenced in any publication
(Suckling J et al., 1994, Excerpta Medica, International Congress Series 1069, pp 375-378). Consequences:
* Keep MIAS in this research workspace. Do not put it in a customer-facing portal, sell models as MIAS-derived commercial datasets, or redistribute the images.
* Confirm with the target journal whether example figures containing MIAS images are acceptable; state the data-availability position (public database, obtain from the owners).
* Ask the owners for written confirmation if IS360 intends any use beyond the paper and internal research.

## Data handling
Public de-identified data; patient ids are pair numbers (mias_pNNN). Source file names are kept only in the manifest. Keep the workspace on an encrypted disk and back it up.

## Study integrity (what the code enforces)
* Partition is deterministic, patient-level and hash-frozen; editing a frozen list stops every later command.
* MS experiments require a frozen protocol; the first run per protocol hash is confirmatory, all later runs are exploratory; every MS access is logged with user, time, code hash and protocol hash.
* Every experiment folder stores a config.json (versions, seeds, code hash, protocol hash, partition hashes).
* Development results are labelled DEVELOPMENT in every table and cannot be mistaken for manuscript results.

## Regulatory
Research use only. Outputs must not be used for clinical decisions. A clinical or commercial product requires separate data, validation and regulatory review.
