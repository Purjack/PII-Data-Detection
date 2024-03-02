import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import BertForTokenClassification, get_linear_schedule_with_warmup
import utils


def pipeline(train_dataloader, test_dataloader, num_epochs):
    # Setup device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration of the model
    model = BertForTokenClassification.from_pretrained(
        'dslim/bert-base-NER',
        _num_labels=len(utils.get_label_2_id()),
        num_labels=len(utils.get_id_2_label()),
        id2label=utils.get_label_2_id(),
        label2id=utils.get_label_2_id(),
        ignore_mismatched_sizes=True
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Total number of training steps
    total_steps = len(train_dataloader) * num_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader):
            # Step 0: Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Step 1: Clear the gradients
            optimizer.zero_grad()

            # Step 2: Forward pass
            outputs = model(**batch)

            # Step 3: Compute loss
            loss = outputs.loss
            train_loss += loss.item()

            # Step 4: Backward pass
            loss.backward()

            # Step 5: Update parameters and take a step using the computed gradient
            optimizer.step()

            # Step 6: Update the learning rate.
            scheduler.step()

        # Compute the average loss over the training data.
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss}")

        model.eval()
        test_loss = 0
        with torch.inference_mode():
            for batch in tqdm(enumerate(test_dataloader)):

                # Send data to target device
                batch = {k: v.to(device) for k, v in batch.items()}

                # 1. Forward pass
                outputs = model(**batch)

                # 2. Calculate the loss/accuracy
                loss = outputs.loss
                test_loss += loss.item()

        # Compute the average loss over the training data.
        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Average test loss: {avg_test_loss}")
