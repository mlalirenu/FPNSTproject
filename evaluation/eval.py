from evaluation.loss import compute_content_loss, compute_style_loss
from evaluation.lpips_utils import compute_lpips_scores
from evaluation.ssim_utils import compute_ssim
import torchvision.transforms as transforms
import os
import datetime

async def evaluate(vgg, stylized_img, content_img, style_image, output_path, device):
    # 1. First, call the LPIPS function using the original PIL images.
    lpips_content, lpips_style = compute_lpips_scores(content_img, style_image, stylized_img)

    # 2. Next, convert the PIL images to tensors for the other loss functions.
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    print("Converting images to tensors and resizing to 512x512...")
    content_img_tensor = transform(content_img).unsqueeze(0)
    style_img_tensor = transform(style_image).unsqueeze(0)
    stylized_img_tensor = transform(stylized_img).unsqueeze(0)
    print(f"Content image tensor size: {content_img_tensor.size()}")
    print(f"Style image tensor size: {style_img_tensor.size()}")
    print(f"Stylized image tensor size: {stylized_img_tensor.size()}")

    # 3. Finally, call the remaining loss functions with the new tensors.
    c_loss, c_similarity = compute_content_loss(vgg, content_img_tensor, stylized_img_tensor, device)
    s_loss, s_similarity = compute_style_loss(vgg, style_img_tensor, stylized_img_tensor, device)
    ssim_content = compute_ssim(content_img_tensor, stylized_img_tensor)

    timestamp = datetime.datetime.now().isoformat()
    output_image_name = os.path.basename(output_path)

    return {
        "Timestamp": timestamp,
        "Image Name": output_image_name,
        "Content Loss (Raw)": c_loss.item(),
        "Content Similarity % (Range: 0-100)": c_similarity,
        "Style Loss (Raw)": s_loss.item(),
        "Style Similarity % (Range: 0-100)": s_similarity,
        "LPIPS Content": lpips_content,
        "LPIPS Style": lpips_style,
        "SSIM Content": ssim_content.item()
    }