import torch
import torchvision
import argparse
import model_builder

if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"

  parser = argparse.ArgumentParser()

  parser.add_argument("--image", help="path to input image")
  parser.add_argument("--model_path", default="models/05_going_modular_script_mode_tinyvgg_model.pth", type=str, help="path to model")

  args = parser.parse_args()

  image_path = args.image
  model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=128,
                                  output_shape=3).to(device)
  model.load_state_dict(torch.load(args.model_path))

  img = torchvision.io.read_image(str(image_path)).type(torch.float32)
  img = img / 255

  data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
  ])

  img = data_transform(img)
  class_names = ["pizza", "steak", "sushi"]

  model.eval()
  with torch.inference_mode():
      img = img.to(device)

      pred_logits = model(img.unsqueeze(dim=0))
      pred_prob = torch.softmax(pred_logits, dim=1)
      pred_label = torch.argmax(pred_prob, dim=1)
      pred_label_class = class_names[pred_label]

  print(f"pred_label_class: {pred_label_class}, pred_prob: {pred_prob.max():.2f}")
