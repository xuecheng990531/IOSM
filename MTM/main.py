import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model.model730_refine import Unet
from tools.metrics2 import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
    # [Previous get_args code remains unchanged]
    parser = argparse.ArgumentParser(description='mtm model')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--load', required=True, help='save model')
    parser.add_argument('--train_batch', type=int, default=12, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size for train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--finetuning', default=False)
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=100, help='number of epochs to save model')
    parser.add_argument('--device', type=str, default='cuda', help='cuda device')
    parser.add_argument('--modelname', type=str, required=True, help='model name')

    args = parser.parse_args()
    print(args)
    return args

def main():
    print("=============> Loading args")
    args = get_args()
    device = args.device

    # Create a unique log directory based on experiment parameters
    log_dir = f"logs/{args.modelname}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_dataset = MattingDataset(
        image_dir=args.train_image_dir,
        trimap_dir=args.train_trimap_dir,
        alpha_dir=args.train_alpha_dir,
        alpha_dir=args.train_mask_dir,
        mode='train',
        seed=42
    )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, num_workers=8, shuffle=True)

    test_dataset = MattingDataset(
        image_dir=args.test_image_dir,
        trimap_dir=args.test_trimap_dir,
        alpha_dir=args.test_alpha_dir,
        alpha_dir=args.test_mask_dir,
        mode='test',
        seed=42
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True)

    model=Unet(backbone_name='resnet18').to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0005, betas=(0.9, 0.999))

    print("============> Start Train ! ...")
    epoch = 0
    loss_ = 0
    while epoch <= args.nEpochs:
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch:{epoch}")
        for index, sample_batched in enumerate(pbar):
            image, alpha, trimap = sample_batched['image'], sample_batched['alpha'], sample_batched['trimap']
            image, alpha, trimap = image.to(device), alpha.to(device), trimap.to(device)

            alpha_pre = model(image,trimap)

            loss1, loss2, loss = fusion_loss(image, alpha, alpha_pre)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()

        writer.add_scalar('train/loss', loss_, epoch)
        loss_ = 0  # Reset loss_ for the next epoch

        if epoch % 10 == 0:
            asad, amse, agrad, aconn = 0, 0, 0, 0
            print('=============> testing')
            model.eval()
            with torch.no_grad():
                pbar_test = tqdm(test_loader, desc='Test')
                for i, sample_batched in enumerate(pbar_test):
                    image, alpha, trimap, test_image_name = sample_batched['image'], sample_batched['alpha'], sample_batched['trimap'], sample_batched['image_name']
                    image, alpha, trimap = image.to(device), alpha.to(device), trimap.to(device)

                    pred_alpha = model(image, trimap)

                    sad, mse, mad = calculate_sad_mse_mad(pred_alpha, alpha, trimap)
                    grad = compute_gradient_whole_image(pred_alpha, alpha)
                    conn = compute_connectivity_loss_whole_image(pred_alpha, alpha)

                    asad += sad
                    amse += mse
                    aconn += conn
                    agrad += grad

                # Log average metrics with unique tags
                writer.add_scalar('test/avg_sad', asad / len(test_loader), epoch)
                writer.add_scalar('test/avg_mse', amse / len(test_loader), epoch)
                writer.add_scalar('test/avg_grad', agrad / len(test_loader), epoch)
                writer.add_scalar('test/avg_conn', aconn / len(test_loader), epoch)
                print('All_model:\n MSE:{:.4f}---SAD:{:.4f}----Grad:{:.4f}---Conn:{:.4f}'.format(amse / len(test_loader), asad / len(test_loader), agrad / len(test_loader), aconn / len(test_loader)))

        epoch += 1

    writer.close()

if __name__ == "__main__":
    main()
