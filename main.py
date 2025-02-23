import argparse
import os
import subprocess
import torch
import torch.distributed as dist
import pprint

from models.stylegan2 import load_stylegan2_model
from models.gan import load_gan_model
from models.decoders.decoder import FlexibleDecoder
from utils.model_utils import clone_model, load_finetuned_model
from utils.gpu import get_gpu_info

from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model
from evaluation.attacks import black_box_attack_binary_based

def initialize_cuda():
    try:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        _ = torch.empty(1).cuda()
        print(f"Discovered {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA initialization failed: {str(e)}")
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Run training or evaluation for the model.")
    parser.add_argument("mode", choices=["train", "eval", "attack"], help="Mode to run the script in")
    
    # Common arguments
    parser.add_argument("--seed_key", type=int, default=2024, help="Seed for the random authentication key")
    parser.add_argument("--stylegan2_url", type=str, default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl", help="URL to load the StyleGAN2 model from")
    parser.add_argument("--self_trained", type=bool, default=False, help="Use a self-trained GAN model")
    parser.add_argument("--self_trained_model_path", type=str, default="generator.pth", help="Path to the self-trained GAN model")
    parser.add_argument("--self_trained_latent_dim", type=int, default=128, help="Latent dim for self-trained GAN")
    parser.add_argument("--saving_path", type=str, default="results", help="Path to save all related results.")

    # Decoder arguments
    parser.add_argument("--num_conv_layers", type=int, default=5, help="Total number of convolutional layers in the model")
    parser.add_argument("--num_pool_layers", type=int, default=5, help="Total number of pooling layers in the model")
    parser.add_argument("--initial_channels", type=int, default=64, help="Initial number of channels for the first convolutional layer")
    parser.add_argument("--num_conv_layers_surr", type=int, default=5, help="Total number of convolutional layers in the model")
    parser.add_argument("--num_pool_layers_surr", type=int, default=5, help="Total number of pooling layers in the model")
    parser.add_argument("--initial_channels_surr", type=int, default=64, help="Initial number of channels for the first convolutional layer")

    # Training arguments
    parser.add_argument("--n_iterations", type=int, default=20000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr_M_hat", type=float, default=1e-4, help="Learning rate for the watermarked model")
    parser.add_argument("--lr_D", type=float, default=1e-4, help="Learning rate for the decoder")
    parser.add_argument("--max_delta", type=float, default=0.01, help="Maximum allowed change per pixel (infinite norm constraint)")
    parser.add_argument("--run_eval", type=bool, default=True, help="Run evaluation function during training")
    parser.add_argument("--convergence_threshold", type=float, default=0.005, help="Threshold between loss_key diff of each 2000 epochs to determine convergence.")
    parser.add_argument("--mask_switch", type=bool, default=False, help="To apply the new masking pipeline")
    parser.add_argument("--mask_threshold", type=float, default=0.2, help="Threshold for mask")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to a checkpoint file to resume training")

    # Evaluation arguments
    parser.add_argument("--num_eval_samples", type=int, default=100, help="Number of images to evaluate")
    parser.add_argument("--watermarked_model_path", type=str, default="watermarked_model.pkl", help="Path to the finetuned watermarked model")
    parser.add_argument("--decoder_model_path", type=str, default="decoder_model.pth", help="Path to the decoder model state dictionary")
    parser.add_argument("--plotting", type=bool, default=False, help="To plot the results of the evaluation")

    # Attack arguments
    parser.add_argument("--attack_method", type=str, default="wb", choices=["wb", "bb"], help="Attack method")
    parser.add_argument("--surrogate_decoder_type", type=str, default="resnet152", help="Type of surrogate decoder to use for bb binary attack")
    parser.add_argument("--train_size", type=int, default=10000, help="training set size for training surrogate decoder")
    parser.add_argument("--image_attack_size", type=int, default=10000, help="size of attack image set")
    parser.add_argument("--best_threshold", type=float, default=1.0, help="best_threshold of the trained decoder (pipeline) as input for the bb attack")

    # DDP arguments
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()

    # Distributed training setup
    if args.mode == "train":
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    else:
        device = initialize_cuda()

    if args.rank == 0:
        print("===== Input Parameters =====")
        pprint.pprint(vars(args))
        print("============================\n")
        get_gpu_info()

    if args.mode == "train":
        k_auth = torch.tensor([0], device=device)

        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
            watermarked_model = clone_model(gan_model).to(device)
            watermarked_model.train()
            decoder = FlexibleDecoder(
                1,
                args.num_conv_layers,
                args.num_pool_layers,
                args.initial_channels,
            ).to(device)
        else:
            local_path = args.stylegan2_url.split('/')[-1]
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            watermarked_model = clone_model(gan_model).to(device)
            watermarked_model.train()
            decoder = FlexibleDecoder(
                1,
                args.num_conv_layers,
                args.num_pool_layers,
                args.initial_channels,
            ).to(device)
            latent_dim = gan_model.z_dim

        # Wrap models with DDP
        watermarked_model = torch.nn.parallel.DistributedDataParallel(
            watermarked_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

        # Optimizers
        optimizer_D = torch.optim.Adagrad(decoder.parameters(), lr=args.lr_D)
        optimizer_M_hat = torch.optim.Adagrad(watermarked_model.parameters(), lr=args.lr_M_hat)

        # Load checkpoint
        start_iter = 0
        initial_loss_history = []
        if args.resume_checkpoint:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
            checkpoint = torch.load(args.resume_checkpoint, map_location=map_location)
            
            watermarked_model.module.load_state_dict(checkpoint['watermarked_model'])
            decoder.module.load_state_dict(checkpoint['decoder'])
            optimizer_M_hat.load_state_dict(checkpoint['optimizer_M_hat'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            
            start_iter = checkpoint['iteration'] + 1
            initial_loss_history = checkpoint['loss_key_history']
            
            if args.rank == 0:
                print(f"Resuming training from iteration {start_iter}")

        train_model(
            gan_model,
            watermarked_model,
            decoder,
            k_auth,
            args.n_iterations,
            latent_dim,
            args.batch_size,
            device,
            args.lr_M_hat,
            args.lr_D,
            args.run_eval,
            args.num_eval_samples,
            args.plotting,
            args.max_delta,
            args.saving_path,
            args.convergence_threshold,
            args.mask_switch,
            args.seed_key,
            args.mask_threshold,
            optimizer_M_hat,
            optimizer_D,
            start_iter,
            initial_loss_history,
            args.rank,
            args.world_size,
        )

    elif args.mode == "eval":
        local_path = args.stylegan2_url.split('/')[-1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim

        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()

        decoder = FlexibleDecoder(
            1,
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)

        k_auth = torch.tensor([0], device=device)
        print(f"k_auth = {k_auth}")
        
        print(args.plotting)
        auc, tpr_at_1_fpr, best_threshold, best_threshold_tpr, loss_lpips_mean, fid_score, mean_max_delta, total_decoder_params = evaluate_model(
            args.num_eval_samples,
            gan_model,
            watermarked_model,
            decoder,
            k_auth,
            device,
            args.plotting,
            latent_dim,
            args.max_delta,
            args.mask_switch,
            args.seed_key,
            args.mask_threshold,
        )

        print(f"AUC score: {auc:.4f}, "
              f"tpr_at_1_fpr: {tpr_at_1_fpr:.4f}, "
              f"best_threshold: {best_threshold:.4f}, "
              f"best_threshold_tpr: {best_threshold_tpr:.4f}, "
              f"loss_lpips_mean: {loss_lpips_mean:.4f}, "
              f"fid_score: {fid_score:.4f}, "
              f"mean_max_delta: {mean_max_delta:.4f}, "
              f"total_decoder_params: {total_decoder_params:.4f}, ")

    elif args.mode == "attack":
        local_path = args.stylegan2_url.split('/')[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim

        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()

        decoder = FlexibleDecoder(
            1,
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)

        from models.decoders.attack_decoder import CombinedModel
        surrogate_decoder = CombinedModel(
            input_channels=3, 
            length_k_auth=1, 
            decoder_total_conv_layers=args.num_conv_layers,
            decoder_total_pool_layers=args.num_pool_layers,
            decoder_initial_channels=args.initial_channels,
        )

        k_auth = torch.tensor([0], device=device)
        print(f"k_auth = {k_auth}")

        if args.attack_method == "bb":
            black_box_attack_binary_based(
                gan_model, 
                watermarked_model, 
                args.max_delta,
                decoder, 
                surrogate_decoder,
                k_auth, 
                latent_dim, 
                device, 
                args.train_size, 
                args.image_attack_size,
                args.best_threshold,
            )


if __name__ == "__main__":
    main()