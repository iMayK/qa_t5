## Code for finetuning t5 on custom QA dataset in SQuAD format
### Details about the python scripts in the repo
* preprocess_raw_data.py -> This file preprocesses raw train.json file in SQuAD format and generates train_data.pt and valid_data.pt
* train_t5_squad.py -> This is main file, it expects train.pt and valid.pt files as input and uses arge.json file to take hyperparameter values and train the t5 model for QA.
* test_t5.py -> This file has the test script to test the pre-trained model.
* test_paraphraser.py -> This file contains code to test t5's paraphrasing model.
