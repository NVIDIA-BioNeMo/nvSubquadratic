"""Test ZydaDataModule."""


from experiments.datamodules.zyda_datamodule import ZydaDataModule

def test_zyda_datamodule():
    """Test ZydaDataModule."""
    # Use streaming=True to avoid downloading the whole dataset
    datamodule = ZydaDataModule(
        dataset_name="Zyphra/Zyda-2",
        tokenizer_name="nvidia/Mistral-NeMo-Minitron-8B-Base",
        batch_size=2,
        max_length=16,
        streaming=True,
    )
    
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)
    
    print("Batch loaded successfully!")
    print("Input IDs shape:", batch["input_ids"].shape)

if __name__ == "__main__":
    test_zyda_datamodule()
