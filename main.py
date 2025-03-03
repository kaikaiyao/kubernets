import argparse 
import os
import torch
import torch.distributed as dist
import pprint
import logging  # Added to use logging.info
import sys

from models.stylegan2 import load_stylegan2_model
from models.gan import load_gan_model
from models.decoder import FlexibleDecoder
from models.attack_combined_model import CombinedModel
from models.model_utils import clone_model, load_finetuned_model, create_z_classifier_model
from utils.gpu import get_gpu_info, initialize_cuda
from utils.file_utils import generate_time_based_string
from utils.logging import setup_logging

from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model
from attack.attacks import attack_label_based

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
    parser.add_argument("--mask_switch_on", action="store_true", help="Enable the new masking pipeline")
    parser.set_defaults(mask_switch_on=False)
    parser.add_argument("--mask_switch_off", dest="mask_switch_on", action="store_false", help="Disable the new masking pipeline")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to a checkpoint file to resume training")

    # Evaluation arguments
    parser.add_argument("--num_eval_samples", type=int, default=100, help="Number of images to evaluate")
    parser.add_argument("--watermarked_model_path", type=str, default="watermarked_model.pkl", help="Path to the finetuned watermarked model")
    parser.add_argument("--decoder_model_path", type=str, default="decoder_model.pth", help="Path to the decoder model state dictionary")
    parser.add_argument("--plotting", type=bool, default=False, help="To plot the results of the evaluation")
    parser.add_argument("--flip_key_type", type=str, default="none", choices=["none", "1", "10", "random"], help="Whether and how to flip the encryption key")
    parser.add_argument("--key_type", type=str, default="csprng", choices=["encryption", "csprng"], help="Type of key generation method (encryption or csprng)")

    # Attack arguments
    parser.add_argument("--attack_type", type=str, default="base_baseline", choices=["base_baseline", "base_secure", "combined_secure", "fixed_secure"], help="Attack type")
    parser.add_argument("--attack_image_type", type=str, default="original_image", choices=["original_image", "random_image", "blurred_image"], help="Type of images to use for attack")
    parser.add_argument("--train_size", type=int, default=100000, help="training set size for training surrogate decoder")
    parser.add_argument("--image_attack_size", type=int, default=10000, help="size of attack image set")
    parser.add_argument("--surrogate_decoder_model_paths", type=str, help="Comma-separated list of paths to pre-trained surrogate decoder models")
    parser.add_argument("--batch_size_surr", type=int, default=16, help="Batch size for training the surrogate decoder")
    parser.add_argument("--num_steps_pgd", type=int, default=1000, help="Number of steps during the attack")
    parser.add_argument("--alpha_values_pgd", type=str, default="0.1,0.5,1.0", help="Alpha values for the attack (comma-separated list of floats)")
    parser.add_argument("--attack_batch_size_pgd", type=int, default=10, help="Batch size for the attack")
    parser.add_argument("--momentum_pgd", type=float, default=0.9, help="Momentum factor for PGD attack")
    parser.add_argument("--finetune_surrogate", action="store_true", help="Fine-tune surrogate with real decoder outputs")

    # DDP arguments
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    # Z-dependent training arguments
    parser.add_argument("--z_dependant_training", action="store_true", help="Enable z-dependent training pipeline")
    parser.set_defaults(z_dependant_training=False)
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for z-dependent training")
    parser.add_argument("--use_privileged_info", action="store_true", help="Enable LUPI (Learning Using Privileged Information) during training")
    parser.set_defaults(use_privileged_info=False)

    # Load command-line arguments
    args = parser.parse_args()

    # Initialize z_classifier to None by default
    z_classifier = None

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
        # If not running in distributed mode, set rank to 0 for logging purposes.
        args.rank = 0

    time_string = generate_time_based_string()

    os.makedirs(args.saving_path, exist_ok=True)

    if args.rank == 0:
        log_file = os.path.join(args.saving_path, f'training_log_{time_string}.txt')
        setup_logging(log_file)

        logging.info("===== Input Parameters =====")
        logging.info(pprint.pformat(vars(args)))
        logging.info("============================\n")
        get_gpu_info()

    if args.mode == "train":
        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
            watermarked_model = clone_model(gan_model).to(device)
            watermarked_model.train()
            decoder = FlexibleDecoder(
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
            output_device=args.local_rank,
            find_unused_parameters=True
        )

        # Optimizers
        optimizer_D = torch.optim.Adam(
            decoder.parameters(), 
            lr=args.lr_D * 50.0,  # Much higher learning rate
            weight_decay=1e-4,    # Add weight decay
            betas=(0.9, 0.999)    # Standard Adam parameters
        )
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
            initial_loss_history = checkpoint['loss_history']
            
            args.n_iterations += start_iter
            if args.rank == 0:
                logging.info(f"Resuming training from iteration {start_iter}")

        # Create the decoder with correct mode
        if args.z_dependant_training:
            # Create a new decoder with z-dependent mode
            decoder = FlexibleDecoder(
                total_conv_layers=args.num_conv_layers,
                total_pool_layers=args.num_pool_layers,
                initial_channels=args.initial_channels,
                num_classes=args.num_classes,
                z_dependant_mode=True,
                latent_dim=latent_dim,
                use_privileged_info=args.use_privileged_info
            ).to(device)
            
            # Wrap the decoder in DDP
            decoder = torch.nn.parallel.DistributedDataParallel(
                decoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )
            
            # Switch to Adam with weight decay for better optimization
            optimizer_D = torch.optim.Adam(
                decoder.parameters(), 
                lr=args.lr_D * 50.0,  # Much higher learning rate
                weight_decay=1e-4,    # Add weight decay
                betas=(0.9, 0.999)    # Standard Adam parameters
            )
            
            if args.rank == 0:
                logging.info(f"Created decoder with z-dependent training mode, {args.num_classes} classes")
                logging.info(f"Using Adam optimizer with lr={args.lr_D * 50.0}, weight_decay=1e-4")
                if args.use_privileged_info:
                    logging.info("LUPI (Learning Using Privileged Information) enabled - using z during training")
            
            # Create the fixed z classifier for latent vector classification
            z_classifier = create_z_classifier_model(
                latent_dim=latent_dim, 
                num_classes=args.num_classes, 
                seed_key=args.seed_key, 
                device=device
            )
            if args.rank == 0:
                logging.info(f"Created fixed z classifier for {args.num_classes} classes")
        else:
            z_classifier = None
            if args.rank == 0:
                logging.info("Using standard binary classification mode")

        train_model(
            time_string,
            gan_model,
            watermarked_model,
            decoder,
            z_classifier,
            args.n_iterations,
            latent_dim,
            args.batch_size,
            device,
            args.run_eval,
            args.num_eval_samples,
            args.plotting,
            args.max_delta,
            args.saving_path,
            args.mask_switch_on,
            args.seed_key,
            optimizer_M_hat,
            optimizer_D,
            start_iter,
            initial_loss_history,
            args.rank,
            args.world_size,
            args.key_type,
            args.z_dependant_training,
            args.num_classes
        )

    elif args.mode == "eval":
        local_path = args.stylegan2_url.split('/')[-1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.self_trained:
            latent_dim = args.self_trained_latent_dim
            gan_model = load_gan_model(args.self_trained_model_path, latent_dim).to(device)
        else:
            # Log the URL being used to load the model
            logging.info(f"Loading StyleGAN2 model from: {args.stylegan2_url}")
            gan_model = load_stylegan2_model(url=args.stylegan2_url, local_path=local_path, device=device)
            latent_dim = gan_model.z_dim
            logging.info(f"StyleGAN2 model loaded with latent dimension: {latent_dim}")

        # Load the watermarked model 
        watermarked_model = load_finetuned_model(args.watermarked_model_path)
        watermarked_model.to(device)
        watermarked_model.eval()
        
        # Log the watermarked model details
        sample_z = torch.randn((1, latent_dim), device=device)
        with torch.no_grad():
            sample_output = watermarked_model(sample_z, None, truncation_psi=1.0, noise_mode="const")
            logging.info(f"Watermarked model loaded, output shape: {sample_output.shape}")
            
            # Also log the original GAN model output shape for comparison
            sample_gan_output = gan_model(sample_z, None, truncation_psi=1.0, noise_mode="const")
            logging.info(f"Original GAN model output shape: {sample_gan_output.shape}")
            
            # Print the URL parameters to verify we're using the correct model
            if not args.self_trained:
                logging.info(f"StyleGAN2 model URL: {args.stylegan2_url}")
                logging.info(f"StyleGAN2 local path: {local_path}")
        
        logging.info(f"Plotting: {args.plotting}")
        
        if args.z_dependant_training:
            # Create the decoder with z-dependent mode for evaluation
            decoder = FlexibleDecoder(
                total_conv_layers=args.num_conv_layers,
                total_pool_layers=args.num_pool_layers,
                initial_channels=args.initial_channels,
                num_classes=args.num_classes,
                z_dependant_mode=True
            ).to(device)
            decoder.load_state_dict(torch.load(args.decoder_model_path))
            
            # Create the z classifier
            z_classifier = create_z_classifier_model(
                latent_dim=latent_dim, 
                num_classes=args.num_classes, 
                seed_key=args.seed_key, 
                device=device
            )
            logging.info(f"Created z classifier for evaluation with {args.num_classes} classes")
        else:
            # Create standard decoder
            decoder = FlexibleDecoder(
                args.num_conv_layers,
                args.num_pool_layers,
                args.initial_channels,
            ).to(device)
            decoder.load_state_dict(torch.load(args.decoder_model_path))
            z_classifier = None

        eval_results = evaluate_model(
            num_images=args.num_eval_samples,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            z_classifier=z_classifier,
            device=device,
            plotting=args.plotting,
            latent_dim=latent_dim,
            max_delta=args.max_delta,
            mask_switch_on=args.mask_switch_on,
            seed_key=args.seed_key,
            flip_key_type=args.flip_key_type,
            key_type=args.key_type,
            z_dependant_training=args.z_dependant_training,
            num_classes=args.num_classes
        )

        auc, tpr_at_1_fpr, lpips_loss, fid_score, mean_max_delta, total_decoder_params = eval_results

        logging.info(f"AUC score: {auc:.4f}, "
                     f"tpr_at_1_fpr: {tpr_at_1_fpr:.4f}, "
                     f"lpips_loss: {lpips_loss:.4f}, "
                     f"fid_score: {fid_score:.4f}, "
                     f"mean_max_delta: {mean_max_delta:.4f}, "
                     f"total_decoder_params: {total_decoder_params:.4f}")

    elif args.mode == "attack":
        local_path = args.stylegan2_url.split('/')[-1]

        # Check if pre-trained surrogate decoder paths are provided
        if args.surrogate_decoder_model_paths is not None:
            train_surrogate = False
            surrogate_paths = args.surrogate_decoder_model_paths.split(',')
        else:
            train_surrogate = True
            surrogate_paths = [None]  # Will create one surrogate decoder

        # Convert alpha_values string to a vector of values
        args.alpha_values_pgd = [float(x) for x in args.alpha_values_pgd.split(',')]

        # Initialize device and DDP if training surrogate decoder
        if train_surrogate:
            dist.init_process_group(backend='nccl', init_method='env://')
            args.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda', args.local_rank)
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()
        else:
            device = initialize_cuda()
            args.rank = 0

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
            args.num_conv_layers,
            args.num_pool_layers,
            args.initial_channels,
        ).to(device)
        decoder.load_state_dict(torch.load(args.decoder_model_path))
        decoder = decoder.to(device)

        # Initialize list to store surrogate decoders
        surrogate_decoders = []

        # Create or load surrogate decoders
        for surrogate_path in surrogate_paths:
            if args.attack_type in ["base_baseline", "base_secure"]:
                surrogate_decoder = FlexibleDecoder(
                    args.num_conv_layers_surr,
                    args.num_pool_layers_surr,
                    args.initial_channels_surr,
                ).to(device)
            elif args.attack_type in ["combined_secure"]:
                surrogate_decoder = CombinedModel(
                    input_channels=3,
                    decoder_total_conv_layers=args.num_conv_layers_surr,
                    decoder_total_pool_layers=args.num_pool_layers_surr,
                    decoder_initial_channels=args.initial_channels_surr,
                    cnn_instance=None,
                    cnn_mode="fresh",
                ).to(device)
            elif args.attack_type in ["fixed_secure"]:
                from key.key import generate_mask_secret_key
                mask_cnn = generate_mask_secret_key(
                    image_shape=(1, 3, 256, 256),
                    seed=2024,
                    device=device,
                    key_type=args.key_type,
                )
                surrogate_decoder = CombinedModel(
                    input_channels=3,
                    decoder_total_conv_layers=args.num_conv_layers_surr,
                    decoder_total_pool_layers=args.num_pool_layers_surr,
                    decoder_initial_channels=args.initial_channels_surr,
                    cnn_instance=mask_cnn,
                    cnn_mode="fixed",
                ).to(device)

            # Wrap surrogate_decoder with DDP if training
            if train_surrogate:
                surrogate_decoder = torch.nn.parallel.DistributedDataParallel(
                    surrogate_decoder,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

            # Load pre-trained weights if provided
            if surrogate_path is not None:
                logging.info(f"Loading pre-trained surrogate decoder from {surrogate_path}")
                state_dict = torch.load(surrogate_path, map_location=device)
                # If DDP is active, load into the module
                if train_surrogate:
                    surrogate_decoder.module.load_state_dict(state_dict)
                else:
                    surrogate_decoder.load_state_dict(state_dict)

            surrogate_decoders.append(surrogate_decoder)

        if train_surrogate:
            logging.info(f"Training {len(surrogate_decoders)} surrogate decoder(s) from scratch.")
        else:
            logging.info(f"Using {len(surrogate_decoders)} pre-trained surrogate decoder(s).")

        attack_label_based(
            attack_type=args.attack_type,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            max_delta=args.max_delta,
            decoder=decoder,
            surrogate_decoders=surrogate_decoders,  # Now passing list of decoders
            latent_dim=latent_dim,
            device=device,
            train_size=args.train_size,
            image_attack_size=args.image_attack_size,
            batch_size=args.batch_size_surr,
            epochs=1,
            attack_batch_size=args.attack_batch_size_pgd,
            num_steps=args.num_steps_pgd,
            alpha_values=args.alpha_values_pgd,
            train_surrogate=train_surrogate,
            finetune_surrogate=args.finetune_surrogate,
            rank=args.rank,
            world_size=args.world_size if train_surrogate else 1,
            momentum=args.momentum_pgd,
            attack_image_type=args.attack_image_type,
            key_type=args.key_type,
        )


if __name__ == "__main__":
    main()
