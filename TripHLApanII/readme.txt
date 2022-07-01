TripHLApan: predicting HLA molecules binding peptides based on triple coding matrix and transfer learning
=============================================

Introduction
------------
TripHLApanII is a pan-specific model for predicting HLA-II-peptide binding.

Prerequisites:
-------------
python3
pytorch
scikit-learn

data preparation:
-------------
1. The peptide and HLA type pairs, as the first column is peptide sequence, the second and third columns are the HLA names of the α and β chains of the HLA molecule, are written to the file 'test.txt';
2. Place the test.txt to '{CODES_PATH}/TripHLApanII/for_prediction/';
3. Then run the command 'python independent_test.py' and the output will be written in the file: '{CODES_PATH}/TripHLApan/for_prediction/test_output.txt'.

Note: You can customize your file source and output location to your needs in the '# configue' block at the end of the file 'independent_test.py'. For a more detailed description of the model, please refer to the paper:
TripHLApan: predicting HLA molecules binding peptides based on triple coding matrix and transfer learning