TripHLApan: predicting HLA Class I molecules binding peptides based on triple coding matrix and transfer learning.
=============================================

Introduction
------------
TripHLApan is a pan-specific model for predicting HLA-I-peptide binding.

Prerequisites:
-------------
python3
pytorch
scikit-learn

data preparation:
-------------
1. The peptide and HLA type pairs, as the first column is peptide sequence, the second column is HLA molecular name, are written to the file 'test.txt';
2. Place the test.txt to '{CODES_PATH}/TripHLApan/for_prediction/';
3. Then run the command 'python independent_test.py' and the output will be written in the '{CODES_PATH}/TripHLApan/for_prediction/outputs/' folder.

Note: You can customize your file source and output location to your needs in the '# configue' block at the end of the file 'independent_test.py'. And you can choose different prediction models based on your data characteristics. See the 'Models /' folder for more model selections. For a more detailed description of the model, please refer to the paper:
