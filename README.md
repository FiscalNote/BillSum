# BillSum

US Bill Summarization Corpus -- material in association with the paper

**Accessing the Dataset**: Download here

For this guide, `PREFIX` refers to the base directory for all the data.

---

For all the experiments described in the paper, the texts were first cleaned using the script `billsum/data_prep/clean_text.py`. Results will be saved into the `PREFIX/clean_final` directory.

## Sumy baselines

Install sumy and run `bill_sum/sumy_baselines.py`

## Preparing labeled data

Run `billsum/data_prep/label_sentences.py` to create labeled dataset.

This script takes each document, splits it into sentences, processes them with Spacy to get useful syntactic features and calculates the Rouge Score relative to the summary.

Outputs for each dataset part will be a pickle file with a dict of (bill_id, sentence data) pairs

```
Bill_id --> [
	('The monthly limitation for each coverage month during the taxable year is an amount equal to the lesser of 50 percent of the amount paid for qualified health insurance for such month, or an amount equal to 112 of in the case of self-only coverage, $1,320, and in the case of family coverage, $3,480. ',
	  [('The ', 186, 'the', '', 'O', 'DET', 'det', 188),
	   ('monthly ', 187, 'monthly', 'DATE', 'B', 'ADJ', 'amod', 188),
	   ('limitation ', 188, 'limitation', '', 'O', 'NOUN', 'nsubj', 197),
	   ...]
	  {'rouge-1': {'f': 0.2545454500809918,
	    'p': 0.3783783783783784,
	    'r': 0.1917808219178082},
	   'rouge-2': {'f': 0.09459459021183367, 'p': 0.14583333333333334, 'r': 0.07},
	   'rouge-l': {'f': 0.16757568176139123,
	    'p': 0.2972972972972973,
	    'r': 0.1506849315068493}}),
	    ...]
```

## Running Bert Models

0. Clone https://github.com/google-research/bert

1. Create train.tsv / test.tsv files with `billsum/bert_helpers/prep_bert.py`. Put these into a separate directory (BERT_DATA_DIR)

2. Download the [Bert-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) model. 

3. Set `$BERT_BASE_DIR` environment variable to point to directory where you downloaded the model

3. Pretrain the Bert Model (run from the cloned bert repo)

```
python create_pretraining_data.py \
  --input_file=BERT_DATA_DIR/all_texts_us_train.txt \
  --output_file=BERT_DATA_DIR/all_texts_us_train.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```
```
python run_pretraining.py \
  --input_file=BERT_DATA_DIR/all_texts_us_train.tfrecord\
  --output_dir=BERT_MODEL_DIR \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

`BERT_MODEL_DIR` is the directory where you want to store your pretrained model.

This will take a while to run. O

4. To train the classifier model run (from bert repo):

``` 
python run_classifier.py   
			--task_name=simple
			--do_train=true   
			--do_predict=true   
			--data_dir=BERT_DATA_DIR   
			--vocab_file=$BERT_BASE_DIR/vocab.txt   
			--bert_config_file=$BERT_BASE_DIR/bert_config.json   
			--init_checkpoint=BERT_MODEL_DIR/model.ckpt-20000   
			--max_seq_length=128  
			--train_batch_size=32   
			--num_train_epochs=3.0   
			--output_dir=BERT_CLASSIFIER_DIR
```

Change `BERT_CLASSIFIER_DIR` to the directory where you want to store the classifier. This script will create a model in the `BERT_CLASSIFIER_DIR` and store the sentence predictions in `BERT_CLASSIFIER_DIR/test_results.tsv`

5. Evaluate results using `bill_sum/bert_helpers/evaluate_bert.py`. Change the prefix variable to point to `BERT_CLASSIFIER_DIR` from above.







