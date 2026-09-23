# Environment status

`requirements.in` lists the direct Python packages imported by the processing,
splitting, rule-building, training, and scoring code. It is deliberately not an
exact lock file: a complete Kaggle `pip freeze` was not retained with the runs.

`evaluation-requirements.txt` records the two versions installed in the isolated
local environment for the final central rescoring. Do not describe either file
as a complete reconstruction of the executed Kaggle image. Pin and test a full
lock file before publishing a reproducibility release.
