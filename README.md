# cll-scl
The implementation code for training SCL method on complementary label datasets.

## Usage

```bash
python scl-train.py --algo=scl-fwd

# or directly run
bash run.sh
```

## Comment:
python scl-train.py --algo=scl-nl --imb_type exp --imb_factor 0.01 --weighting 1 --mixup true --cl_aug true

##
usage: scl-train.py [-h] [--algo {scl-exp,scl-nl,scl-fwd}] [--dataset_name {cifar10,cifar20}] [--model {resnet18,m-resnet18}] [--lr LR] [--seed SEED] [--data_aug DATA_AUG]
                    [--max_train_samples MAX_TRAIN_SAMPLES] [--evaluate_step EVALUATE_STEP] [--n_epoch N_EPOCH] [--batch_size BATCH_SIZE] [--multi_label] [--imb_type IMB_TYPE]
                    [--imb_factor IMB_FACTOR] [--weighting WEIGHTING] [--mixup MIXUP] [--intra_class INTRA_CLASS] [--cl_aug CL_AUG]

