from torch.utils.data import Dataloader

# KEEP THE FOLLOWING IN MIND
# the Dataset retrieves our dataset's features and labels one sample at a time. while training a model, we typically want to pass samples in "minibatches",
# reshuffle the data at every epoch to reduce model overfitting, and use python's multiprocessing to speed up data retrieval.
train_dataloader=Dataloader(training_data,batch_size=64,shuffle=True)
# TEST AS WELL

train_features,train_lables=next(iter(train_dataloader))
# we have load the dataset into the Dataloader and can iterate through the dataset as needed. each iteration above returns a batch of train_features and train_labels
# because we specified shuffle=True, after we iterate over all batches the data is shuffled.