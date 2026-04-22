import os
import argparse
from ast import parse
from training import train_encoder_decoder_embeddings, train_encoder_decoder_multidata_embeddings, train_ok_network_embeddings, train_ok_network_multidata_embeddings
from video_preprocessing import computeOpticalFlow, create_data_blobs
from embeddings_cluster_explore import evaluate_model, evaluate_model_multidata, plot_umap_clusters, plot_umap_clusters_multidata
from test_oknet_embeddings import evaluate_oknet_model, store_oknet_embeddings_in_dict, plot_oknet_umap_clusters, evaluate_model_superuser
from neural_networks import encoderDecoder, OKNetV4
import pandas as pd
import numpy as np
import torch
import pickle

import warnings
warnings.filterwarnings('ignore')

def main() -> None:
    parser = argparse.ArgumentParser(description='Argument parser to pre-process, train, evaluate and plot results from an encoder-decoder architecture.')
    parser.add_argument('--mode', metavar='--mode', type=str, required=True)
    
    # Training arguments
    parser.add_argument('--lr', metavar='--lr', type=str)
    parser.add_argument('--num_epochs', metavar='--num_epochs', type=str)
    parser.add_argument('--blobs_folder_path', metavar='--blobs_folder_path', type=str)
    parser.add_argument('--weights_save_path', metavar='--weights_save_path', type=str)
    parser.add_argument('--weights_save_folder', metavar='--weights_save_folder', type=str)
    parser.add_argument('--time', default=False, action='store_true')
    parser.add_argument('--no-time', dest='time', action='store_false')
    #parser.add_argument('--time', action=argparse.BooleanOptionalAction)

    
    # Preprocess arguments
    parser.add_argument('--source_directory', metavar='--source_directory', type=str)
    parser.add_argument('--resized_video_directory', metavar='--resized_video_directory', type=str)
    parser.add_argument('--destination_directory', metavar='--destination_directory', type=str)
    parser.add_argument('--resize_dim', metavar='--resize_dim', type=list, nargs=2)

    parser.add_argument('--optical_flow_path', metavar='--optical_flow_path', type=str)
    parser.add_argument('--transcriptions_path', metavar='--transcriptions_path', type=str)
    parser.add_argument('--kinematics_path', metavar='--kinematics_path', type=str)
    parser.add_argument('--frames_per_blob', metavar='--frames_per_blob', type=str)
    parser.add_argument('--blobs_path', metavar='--blobs_path', type=str)
    parser.add_argument('--spacing', metavar='--spacing', type=str)

    # Eval 
    parser.add_argument('--model_dim', metavar='--model_dim', type=str)
    parser.add_argument('--embedding_path', metavar='--embedding_path', type=str)
    parser.add_argument('--plot_save_path', metavar='--plot_save_path', type=str)
    parser.add_argument('--experimental_setup_path', metavar='--experimental_setup_path', type=str)
    parser.add_argument('--labels_store_path', metavar='--labels_store_path', type=str)
    
    # Umaps
    parser.add_argument('--plots_store_path', metavar='--plot_store_path', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_path = args.blobs_folder_path
            if 'knot' or 'tying' in blobs_folder_path.lower():
                dataset_name = 'Knot_Tying'
            elif 'needle' or 'passing' in blobs_folder_path.lower():
                dataset_name = 'Needle_Passing'
            elif 'suturing' in blobs_folder_path.lower():
                dataset_name = 'Suturing'
            else:
                dataset_name = 'dataset'
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        try:
            time = args.time
        except:
            time = False

        train_encoder_decoder_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name, time=time)
    
    elif args.mode == 'oknet_train':
        print("training oknet")
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_path = args.blobs_folder_path
            if 'knot' or 'tying' in blobs_folder_path.lower():
                dataset_name = 'Knot_Tying'
            elif 'needle' or 'passing' in blobs_folder_path.lower():
                dataset_name = 'Needle_Passing'
            elif 'suturing' in blobs_folder_path.lower():
                dataset_name = 'Suturing'
            else:
                dataset_name = 'dataset'
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        try:
            time = args.time
        except:
            time = True

        train_ok_network_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name, time=time)
    
    elif args.mode == 'multidata_train':
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_paths_list = args.blobs_folder_path
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        train_encoder_decoder_multidata_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_paths_list = blobs_folder_paths_list, weights_save_path = weights_save_path, weight_decay = weight_decay)
    
    elif args.mode == 'multidata_oknet_train':
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_paths_list = args.blobs_folder_path
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        train_ok_network_multidata_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_paths_list = blobs_folder_paths_list, weights_save_path = weights_save_path, weight_decay = weight_decay)
    
    elif args.mode == 'optical_flow':
        try:
            source_directory = args.source_directory
        except Exception as e:
            print(e)
        
        try:
            resized_video_directory = args.resized_video_directory
        except Exception as e:
            print(e)
        
        try:
            destination_directory = args.destination_directory
        except Exception as e:
            print(e)
        
        try:
            resize_dim = tuple(args.resize_dim)
        except:
            resize_dim = (320, 240)
        optical_flow_compute = computeOpticalFlow(source_directory = source_directory, resized_video_directory = resized_video_directory, destination_directory = destination_directory, resize_dim = resize_dim)
        optical_flow_compute.run()
    
    elif args.mode == 'data_blobs':
        try:
            optical_flow_folder_path = args.optical_flow_path
        except Exception as e:
            print(e)
        
        try:
            transcriptions_folder_path = args.transcriptions_path
        except Exception as e:
            print(e)
        
        try:
            kinematics_folder_path = args.kinematics_path
        except Exception as e:
            print(e)
        
        try:
            num_frames_per_blob = int(args.frames_per_blob)
        except:
            num_frames_per_blob = 25
        
        try:
            blobs_save_folder_path = args.blobs_path
        except Exception as e:
            print(e)

        try:
            spacing = int(args.spacing)
        except:
            spacing = 2
        create_data_blobs(optical_flow_folder_path = optical_flow_folder_path, transcriptions_folder_path = transcriptions_folder_path, kinematics_folder_path = kinematics_folder_path, num_frames_per_blob = num_frames_per_blob, blobs_save_folder_path = blobs_save_folder_path, spacing = spacing)
    
    elif args.mode == 'eval':
        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        
        try:
            model_dim = int(args.model_dim)
        except:
            model_dim = 2048
            
        model = encoderDecoder(embedding_dim = model_dim)
        model.load_state_dict(torch.load(weights_save_path))
        evaluate_model(blobs_folder_path = blobs_folder_path, model = model, num_clusters = 10, save_embeddings = False)
        
    elif args.mode == 'oknet_eval':
        try:
            embedding_path = args.embedding_path
        except:
            embedding_path = '../embeddings/oknet_embeddings.p'
        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        
        try:
            model_dim = int(args.model_dim)
        except:
            model_dim = 2048
            
        net = OKNetV4(embedding_dim=model_dim)
        #net.load_state_dict(torch.load(weights_save_path))
            
        if not os.path.exists(embedding_path):
            embedding_dict = store_oknet_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=net)
            pickle.dump(embedding_dict, open(embedding_path, "wb"))
        else:
            embedding_dict = pickle.load(open(embedding_path, "rb"))
            
        
        evaluate_oknet_model(embedding_dict, blobs_folder_path=blobs_folder_path, model=net, num_clusters=10, save_embeddings=False)
        
    elif args.mode == 'eval_superuser':
        try:
            embedding_path = args.embedding_path
        except:
            raise RuntimeError('Please specify the embeddings')
        
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)

        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            transcriptions_folder_path = args.transcriptions_path
        except Exception as e:
            print(e)
        
        try:
            experimental_setup_path = args.experimental_setup_path
        except Exception as e:
            print(e)
            
        net = OKNetV4(embedding_dim=2048)
        net.load_state_dict(torch.load(weights_save_path))
        
        if not os.path.exists(embedding_path):
            raise RuntimeError('Please enter valid embedding path')
        
        embedding_dict = embedding_dict = pickle.load(open(embedding_path, "rb"))
        evaluate_model_superuser(embedding_dict=embedding_dict, blobs_folder_path=blobs_folder_path, model=net, transcriptions_path=transcriptions_folder_path, experimental_setup_path=experimental_setup_path)
        
    elif args.mode == 'eval_onetrialout_all':
        
        try:
            embedding_path = args.embedding_path
        except:
            raise RuntimeError('Please specify the embeddings')
        
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)

        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            transcriptions_folder_path = args.transcriptions_path
        except Exception as e:
            print(e)
        
        try:
            experimental_setup_path = args.experimental_setup_path
        except Exception as e:
            print(e)
            
        net = OKNetV4(embedding_dim=2048)
        #net.load_state_dict(torch.load(weights_save_path))
        
        if not os.path.exists(embedding_path):
            raise RuntimeError('Please enter valid embedding path')
        
        embedding_dict = embedding_dict = pickle.load(open(embedding_path, "rb"))
        
        accuracies = []
        i=1
        itrs = os.listdir(experimental_setup_path)
        itrs = list(filter(lambda x: '.DS_Store' not in x, itrs))
        itrs.sort(key=lambda x:x.split('_')[0])

        for split in itrs:
            print(f'processing {split} --> split {i} of {len(itrs)}')
            split_path = os.path.join(experimental_setup_path, split)
            split_acc = evaluate_model_superuser(embedding_dict=embedding_dict, blobs_folder_path=blobs_folder_path, model=net, transcriptions_path=transcriptions_folder_path, experimental_setup_path=split_path)
            accuracies.append(split_acc)
            print(f'current split accuracy --> {split_acc}')
            print(f'current average accuracy --> {np.mean(accuracies)}')
            print(f'current accuracies list {accuracies}')
            i+=1
            
        print(f'Average over all splits: {np.mean(accuracies)}')
        
    elif args.mode == 'multimodel_oknet_eval':
        # Finding the best model out of all of our weights
        try:
            embedding_path = args.embedding_path
        except:
            embedding_path = '../embeddings/oknet_embeddings.p'
        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            weights_save_folder = args.weights_save_folder
        except Exception as e:
            print(e)
        
        try:
            model_dim = int(args.model_dim)
        except:
            model_dim = 2048
            
        df = pd.DataFrame(columns=['model', 'gesture_acc', 'skill_acc', 'user_acc'])
            
        net = OKNetV4(embedding_dim=model_dim)
        #net.load_state_dict(torch.load(weights_save_path))
            
        for weights_save_path in os.listdir(weights_save_folder):
            net.load_state_dict(torch.load(os.path.join(weights_save_folder, weights_save_path)))
            
            #embedding_dict = pickle.load(open(embedding_path, "rb"))
            embedding_dict = store_oknet_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=net)
            pickle.dump(embedding_dict, open(embedding_path, "wb"))
            
            accs = evaluate_oknet_model(embedding_dict, blobs_folder_path=blobs_folder_path, model=net, num_clusters=10, save_embeddings=False)
            df.loc[len(df), :] = [weights_save_path] + accs
            df.to_csv('../Misc/accuracies.csv')
        
        #evaluate_oknet_model(embedding_dict, blobs_folder_path=blobs_folder_path, model=net, num_clusters=10, save_embeddings=False)
        
    elif args.mode == 'oknet_umaps':
        try:
            embedding_path = args.embedding_path
        except:
            embedding_path = '../embeddings/oknet_embeddings.p'
        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)
        
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        
        try:
            plots_store_path = args.plots_store_path
        except Exception as e:
            print(e)
            
        net = OKNetV4(embedding_dim=2048)
        net.load_state_dict(torch.load(weights_save_path))
            
        if not os.path.exists(embedding_path):
            embedding_dict = store_oknet_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=net)
            pickle.dump(embedding_dict, open(embedding_path, "wb"))
        else:
            embedding_dict = pickle.load(open(embedding_path, "rb"))
            
        
        plot_oknet_umap_clusters(embedding_dict=embedding_dict, blobs_folder_path=blobs_folder_path, model=net, plot_store_path=plots_store_path)
    elif args.mode == 'vit_preprocess':
        # New ViT preprocessing mode
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        
        try:
            from precompute_raft import main as precompute_raft_main
            from precompute_eeg_rdms import main as precompute_eeg_rdms_main
            
            # Run RAFT preprocessing if requested
            if args.optical_flow_path and args.source_directory:
                import subprocess
                cmd = [
                    'python', str(Path(__file__).parent.parent / 'scripts' / 'precompute_raft.py'),
                    '--video_dir', args.source_directory,
                    '--output_dir', args.optical_flow_path,
                    '--raft_weights', args.weights_save_path if args.weights_save_path else 'raft_weights.pth'
                ]
                subprocess.run(cmd)
            
            # Run EEG RDM preprocessing if requested
            if args.experimental_setup_path:  # Reuse this arg for EEG cache dir
                import subprocess
                cmd = [
                    'python', str(Path(__file__).parent.parent / 'scripts' / 'precompute_eeg_rdms.py'),
                    '--data_root', args.source_directory if args.source_directory else '.',
                    '--task', args.transcriptions_path if args.transcriptions_path else 'Knot_Tying',
                    '--cache_dir', args.experimental_setup_path
                ]
                subprocess.run(cmd)
        except Exception as e:
            print(f"Error in vit_preprocess: {e}")
            print("Make sure scripts/precompute_raft.py and scripts/precompute_eeg_rdms.py are available")
    
    elif args.mode == 'vit_train':
        # New ViT training mode
        import sys
        from pathlib import Path
        import importlib.util
        
        try:
            # Import using file path to avoid conflict with local training.py
            src_path = Path(__file__).parent.parent / 'src'
            sys.path.insert(0, str(src_path))
            
            # Use importlib to explicitly load from src/training with proper package context
            spec = importlib.util.spec_from_file_location(
                "training.train_vit_system",
                src_path / "training" / "train_vit_system.py",
                submodule_search_locations=[str(src_path)]
            )
            train_module = importlib.util.module_from_spec(spec)
            # Set __package__ to allow relative imports to work
            train_module.__package__ = 'training'
            train_module.__name__ = 'training.train_vit_system'
            spec.loader.exec_module(train_module)
            vit_train_main = train_module.main
            
            import yaml
            
            # Create config from args or load from file
            config_path = args.weights_save_path if args.weights_save_path and args.weights_save_path.endswith('.yaml') else None
            if not config_path:
                # Use default config based on brain mode
                brain_mode = getattr(args, 'brain_mode', 'none') if hasattr(args, 'brain_mode') else 'none'
                config_path = str(Path(__file__).parent.parent / 'src' / 'configs' / f'{brain_mode}.yaml')
            
            # Override with command line args
            sys.argv = [
                'train_vit_system.py',
                '--config', config_path,
                '--data_root', args.source_directory if args.source_directory else '.',
                '--task', args.transcriptions_path if args.transcriptions_path else 'Knot_Tying',
                '--output_dir', args.weights_save_folder if args.weights_save_folder else 'checkpoints'
            ]
            
            vit_train_main()
        except Exception as e:
            print(f"Error in vit_train: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'vit_eval':
        # New ViT evaluation mode
        import sys
        from pathlib import Path
        import importlib.util
        
        try:
            src_path = Path(__file__).parent.parent / 'src'
            sys.path.insert(0, str(src_path))
            
            # Import modules avoiding conflict with local training.py
            from eval.metrics import compute_kinematics_metrics, compute_gesture_metrics, compute_skill_metrics
            from data import JIGSAWSViTDataset
            
            # Import train_vit_system using importlib to avoid conflict
            spec = importlib.util.spec_from_file_location(
                "training.train_vit_system",
                src_path / "training" / "train_vit_system.py",
                submodule_search_locations=[str(src_path)]
            )
            train_module = importlib.util.module_from_spec(spec)
            # Set __package__ to allow relative imports to work
            train_module.__package__ = 'training'
            train_module.__name__ = 'training.train_vit_system'
            spec.loader.exec_module(train_module)
            EEGInformedViTModel = train_module.EEGInformedViTModel
            from torch.utils.data import DataLoader
            import torch
            import yaml
            
            # Load model
            checkpoint_path = args.weights_save_path
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            model = EEGInformedViTModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            # Create dataset
            dataset = JIGSAWSViTDataset(
                data_root=args.source_directory if args.source_directory else '.',
                task=args.transcriptions_path if args.transcriptions_path else 'Knot_Tying',
                mode='val'
            )
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            # Evaluate
            all_metrics = {}
            with torch.no_grad():
                for batch in dataloader:
                    rgb = batch['rgb'].to(device)
                    kinematics = batch['kinematics'].to(device)
                    gesture_labels = batch['gesture_label'].to(device)
                    skill_labels = batch['skill_label'].to(device)
                    
                    outputs = model(rgb)
                    
                    # Compute metrics
                    kin_metrics = compute_kinematics_metrics(outputs['kinematics'], kinematics)
                    gest_metrics = compute_gesture_metrics(outputs['gesture_logits'], gesture_labels)
                    skill_metrics = compute_skill_metrics(outputs['skill_logits'], skill_labels)
                    
                    # Accumulate
                    for k, v in {**kin_metrics, **gest_metrics, **skill_metrics}.items():
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(v)
            
            # Average metrics
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            print("Evaluation metrics:", avg_metrics)
            
        except Exception as e:
            print(f"Error in vit_eval: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'vit_inference':
        # New ViT inference mode
        import sys
        from pathlib import Path
        import importlib.util
        
        try:
            src_path = Path(__file__).parent.parent / 'src'
            sys.path.insert(0, str(src_path))
            
            # Import train_vit_system using importlib to avoid conflict
            spec = importlib.util.spec_from_file_location(
                "training.train_vit_system",
                src_path / "training" / "train_vit_system.py",
                submodule_search_locations=[str(src_path)]
            )
            train_module = importlib.util.module_from_spec(spec)
            # Set __package__ to allow relative imports to work
            train_module.__package__ = 'training'
            train_module.__name__ = 'training.train_vit_system'
            spec.loader.exec_module(train_module)
            EEGInformedViTModel = train_module.EEGInformedViTModel
            
            from eval.postprocess import postprocess_kinematics
            import torch
            import cv2
            import yaml
            
            # Load model
            checkpoint_path = args.weights_save_path
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            model = EEGInformedViTModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            # Load video
            video_path = args.source_directory
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            # Process frames
            frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
            frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
            
            with torch.no_grad():
                outputs = model(frames_tensor)
                kinematics = outputs['kinematics']
            
            # Post-process
            kinematics_processed = postprocess_kinematics(kinematics[0])
            
            # Save results
            output_path = args.destination_directory if args.destination_directory else 'inference_output.npy'
            np.save(output_path, kinematics_processed.cpu().numpy())
            print(f"Saved predictions to {output_path}")
            
        except Exception as e:
            print(f"Error in vit_inference: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print('Mode is not recognized. Options are optical_flow, data_blobs, train, multidata_train, eval, vit_preprocess, vit_train, vit_eval, or vit_inference')



if __name__ == '__main__':
    main()
