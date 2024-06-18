"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"

  parser = argparse.ArgumentParser()

  parser.add_argument("--num_epochs", default=5, type=int, help="the number of training epochs")
  parser.add_argument("--batch_size", default=32,type=int, help="the number of images in one batch")
  parser.add_argument("--hidden_units", default=10, type=int, help="the number of hidden units")
  parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
  parser.add_argument("--train_dir", default="data/pizza_steak_sushi/train", type=str, help="directory file path to training data")
  parser.add_argument("--test_dir", default="data/pizza_steak_sushi/test", type=str, help="directory file path to testing data")

  args = parser.parse_args()

  NUM_EPOCHS = args.num_epochs
  BATCH_SIZE = args.batch_size
  HIDDEN_UNITS = args.hidden_units
  LEARNING_RATE = args.learning_rate
  train_dir = args.train_dir
  test_dir = args.test_dir

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=HIDDEN_UNITS,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  target_dir="models",
                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
