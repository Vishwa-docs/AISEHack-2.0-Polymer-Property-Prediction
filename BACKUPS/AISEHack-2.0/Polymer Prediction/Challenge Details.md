Can your machine learning model help to uncover the secrets of polymers? In this competition, you're tasked with predicting the properties of polymers to speed up the development of new materials. Your contributions will help researchers innovate faster, paving the way for more sustainable and biocompatible materials that can positively impact our planet.

Your mission is to predict a polymer's real-world performance directly from its chemical structure. You'll be provided with a polymer's structure as a simple text string (SMILES), and your challenge is to build a model that can accurately forecast two different polymer key metrics. This includes predicting its glass transition temperature (Tg) and chain band gap (Egc). While Tg is a measure of thermal stability of polymers, Egc indicate their electrical performance.

Evaluation
link
keyboard_arrow_up
The evaluation metric for this competition is the mean coefficient of determination R2 across the two targets: Tg and Egc.


where R² is defined as:


where yᵢ = ground truth, ŷᵢ = predictions, and ȳ = mean of ground truth values.

Submission File
link
keyboard_arrow_up
The submission file for this competition must be a csv file. For each id in the test set, you must predict based on the "target_type". The file should contain a header and have the following format.

id  target
1   220
2   2.3
3   110
4   70

Dataset Description
The training dataset contains a total of 6171 data points combining both properties, where:

Tg values are reported in °C
Egc values are reported in eV
The "target_type" column can be used to identify the two different target properties (Tg and Egc).

Leaderboard and Qualification

Final rankings will be determined using a hidden private test set, which will serve as the official evaluation dataset. Performance on this private set will be used to determine the teams that advance to the next stage of the competition.

To provide participants with feedback during the competition, a subset of the test data will be used to compute the public leaderboard score. The public and private test sets contain 1543 and 2572 polymers, respectively.

Submission Format and Baseline Model

An example submission file is provided to illustrate the required prediction format.

To help participants get started, we also provide a baseline notebook demonstrating a complete machine learning workflow for this task. The baseline model uses the open-source RDKit library to extract molecular descriptors from polymer SMILES representations, applies basic feature engineering, and trains a predictive model for both target properties.

Participants are encouraged to build upon and improve this baseline approach to achieve higher predictive performance.

Files

train.csv - the training set

smiles : SMILES representation of polymer structures
target : Property value( Either Tg/Egc)
target_type : Whether the reported value belongs to Tg or Egc
test.csv - The test set

id : Unique identifier for each polymer
smiles : SMILES representation of polymer structures
target_type : Whether the reported value belongs to Tg or Egc
sample_submission.csv - a sample submission file in the correct format

base_line_model.ipynb – A baseline Ridge Regression model demonstrating cheminformatics feature generation using the RDKit library, model development, and creation of the final submission.csv file.
