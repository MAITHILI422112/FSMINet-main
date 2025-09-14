from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np

from data_loader import Rescale, ToTensor, SalObjDataset
from model import FSMINet

# ---------------- Metrics ----------------
def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta2=0.3*0.3):
    pred_bin = (pred >= 0.5).astype(np.uint8)
    gt_bin = (gt >= 0.5).astype(np.uint8)

    tp = np.sum(pred_bin * gt_bin)
    prec = tp / (np.sum(pred_bin) + 1e-8)
    rec  = tp / (np.sum(gt_bin) + 1e-8)

    return (1+beta2)*prec*rec / (beta2*prec + rec + 1e-8)

def s_measure(pred, gt):
    # Simplified structure measure (Fan et al. 2017)
    pred_mean = np.mean(pred)
    gt_mean   = np.mean(gt)

    alpha = 0.5
    # Object-aware similarity
    So = 1 - np.abs(pred_mean - gt_mean)
    # Region-aware similarity (rough, not full implementation)
    Sr = np.mean(1 - np.abs(pred - gt))
    return alpha*So + (1-alpha)*Sr

# ---------------- Prediction Normalization ----------------
def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)
    return (x-MIN)/(MAX-MIN)

# ---------------- Save Output ----------------
def save_output(image_name, pred, d_dir):
    predict = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray((predict*255).astype(np.uint8)).convert('L')
    img_name = os.path.basename(image_name)
    image = io.imread(image_name)
    im = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    save_path = os.path.join(d_dir, img_name.replace('.jpg', '.png'))
    im.save(save_path)
    return save_path

# ---------------- Main ----------------
if __name__ == '__main__':
    # Paths
    image_dir = "/kaggle/input/eorssd/test-images/"
    gt_dir = "/kaggle/input/eorssd/test-labels/"
    prediction_dir = "/kaggle/working/test_results/"
    os.makedirs(prediction_dir, exist_ok=True)
    model_dir = "/kaggle/working/FSMINet-main/model_save/FSMINet.pth"

    img_name_list = glob.glob(image_dir + '*.jpg')

    # Data
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([Rescale(384), ToTensor(flag=0)])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Model
    print("...load FSMINet...")
    net = FSMINet()
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # Metrics accumulators
    mae_list, f_list, s_list = [], [], []

    # Run test loop
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader), total=len(test_salobj_dataloader)):
        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d0, d1, d2, d3, d4, d5 = net(inputs_test)
        pred = normPRED(d0[:,0,:,:])

        # Save prediction
        pred_path = save_output(img_name_list[i_test], pred, prediction_dir)

        # ---- Load GT and prediction for metrics ----
        gt_name = os.path.basename(img_name_list[i_test]).replace('.jpg', '.png')
        gt_path = os.path.join(gt_dir, gt_name)
        if os.path.exists(gt_path):
            gt = io.imread(gt_path, as_gray=True) / 255.0
            pr = io.imread(pred_path, as_gray=True) / 255.0

            if pr.shape != gt.shape:
                pr = np.array(Image.fromarray((pr*255).astype(np.uint8)).resize(gt.shape[::-1]))/255.0

            mae_list.append(mae(pr, gt))
            f_list.append(f_measure(pr, gt))
            s_list.append(s_measure(pr, gt))

        del d0, d1, d2, d3, d4, d5

    # ---- Print final results ----
    print("\n=== Evaluation Results ===")
    print(f"MAE: {np.mean(mae_list):.4f}")
    print(f"F-measure: {np.mean(f_list):.4f}")
    print(f"S-measure: {np.mean(s_list):.4f}")
