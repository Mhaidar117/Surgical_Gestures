from black import out
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
from joblib import dump, load
import re
import umap
import traceback
import sys
import time

from neural_networks import OKNetV4



def store_oknet_embeddings_in_dict(blobs_folder_path: str, model: OKNetV4) -> dict:
    blobs_folder = os.listdir(blobs_folder_path)
    blobs_folder = list(filter(lambda x: ".DS_Store" not in x, blobs_folder))
    blobs_folder.sort(key=lambda x: int(x.split("_")[1]))

    opt_embeddings_list = []
    kin_embeddings_list = []
    raw_kin_list = []
    gestures_list = []
    user_list = []
    skill_dict = {"B": 0, "C": 1, "D": 2, "E": 2, "F": 1, "G": 0, "H": 0, "I": 0}
    skill_list = []
    file_list = []

    opt_encoder = model.optical_flow_rnn_stream
    kin_encoder = model.kinematics_rnn_stream
    opt_encoder.eval()
    kin_encoder.eval()

    for file in blobs_folder:
        print("Processing file {}".format(file))

        curr_path = os.path.join(blobs_folder_path, file)
        opt_blob, kin_blob = pickle.load(open(curr_path, "rb"))
        
        print(opt_blob.size())
        
        try:
            timesteps = opt_blob.size()[0] // 2
            
            #opt_blob = opt_blob.view(1, timesteps, 2, 240, 320) #uncomment for the original way of doing it 
            opt_blob = opt_blob.split(2)
            opt_blob = torch.stack(opt_blob, dim=0)
            opt_blob = opt_blob.unsqueeze(0)

            print(opt_blob.size())
            kin_blob = kin_blob.view(1, timesteps, 1, 76)


            opt_out, _ = opt_encoder(opt_blob)
            kin_out = kin_encoder(kin_blob)
            # out = model(curr_blob)
            opt_out = opt_out.cpu().detach().data.numpy()
            kin_out = kin_out.cpu().detach().data.numpy()
            kin_raw = kin_blob.cpu().data.numpy().reshape(1, -1)
            opt_embeddings_list.append(opt_out)
            kin_embeddings_list.append(kin_out)
            raw_kin_list.append(kin_raw)

            file_list.append(file)
            file = file.split("_")
            gestures_list.append(file[-1].split(".")[0])
            user_list.append(file[3][0])
            skill_list.append(skill_dict[file[3][0]])
        except Exception as e:
            print(f"exception in file {file}")
            print(e)
            pass

    final_dict = {
        "gesture": gestures_list,
        "user": user_list,
        "skill": skill_list,
        "opt_embeddings": opt_embeddings_list,
        "kin_embeddings": kin_embeddings_list,
        "kin_raw": raw_kin_list,
        "file_list": file_list,
    }

    return final_dict


def cluster_oknet_statistics(
    embedding_dict, blobs_folder_path: str, model: OKNetV4, num_clusters: int
) -> pd.DataFrame:
    # results_dict = store_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=model)
    results_dict = embedding_dict
    k_means = KMeans(n_clusters=num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict["opt_embeddings"]).reshape(-1, 2048))
    results_dict["cluster_indices"] = cluster_indices
    df = pd.DataFrame(results_dict)
    return df


def cluster_statistics_multidata(
    blobs_folder_paths_list: List[str], model: OKNetV4, num_clusters: int
) -> pd.DataFrame:
    results_dict = {"gesture": [], "user": [], "skill": [], "embeddings": [], "task": []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_oknet_embeddings_in_dict(blobs_folder_path=path, model=model)
        # import pdb; pdb.set_trace()
        temp_results_dict["task"] = [idx] * len(temp_results_dict["skill"])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)
    k_means = KMeans(n_clusters=num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict["embeddings"]).reshape(-1, 2048))
    results_dict["cluster_indices"] = cluster_indices
    df = pd.DataFrame(results_dict)
    return df


def evaluate_oknet_model(
    embedding_dict, blobs_folder_path: str, model: OKNetV4, num_clusters: int, save_embeddings: bool
) -> None:
    df = cluster_oknet_statistics(embedding_dict, blobs_folder_path=blobs_folder_path, model=model, num_clusters=num_clusters)
    if save_embeddings:
        print("Saving dataframe.")
        df.to_pickle("./df.p")
    y = df["gesture"].values.ravel()
    
    y_skill = df["skill"].values.ravel()
    y_user = df["user"].values.ravel()
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    
    # Encode skills
    le_skill = LabelEncoder()
    le_skill.fit(y_skill)
    y_skill_encoded = le_skill.transform(y_skill)
    
    # Encode users
    le_user = LabelEncoder()
    le_user.fit(y_user)
    y_user_encoded = le_user.transform(y_user)

    opt_X = [np.array(v) for v in df["opt_embeddings"]]
    kin_X = [np.array(v) for v in df["kin_embeddings"]]
    kin_raw_X = [np.array(v) for v in df["kin_raw"]]
    opt_X = np.array(opt_X).reshape(-1, 2048)
    kin_X = np.array(kin_X).reshape(-1, 2048)  # full: 2048, reduced: 682
    #kin_raw_X = np.array(kin_raw_X).reshape(-1, 1900)
    #X = opt_X
    X = kin_X
    #X = np.hstack((opt_X, kin_X))
    # X = kin_raw_X
    classifier = XGBClassifier(n_estimators=1000)
    
    accs = []
    y_dict = {'gesture': y_encoded, 'skill': y_skill_encoded, 'user': y_user_encoded}
    for y in y_dict.keys():
        X_train, X_test, y_train, y_test = train_test_split(X, y_dict[y], random_state=8765)
        print(f"Training XGBClassifier... for {y}")
        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_train)
        y_hat_test = classifier.predict(X_test)

        print("Training set classification report.")
        print(classification_report(y_train, y_hat))

        print("Test set classification report.")
        print(classification_report(y_test, y_hat_test))
        
        accs.append(accuracy_score(y_test, y_hat_test))
        
    return accs


def plot_oknet_umap_clusters(embedding_dict, blobs_folder_path: str, model: OKNetV4, plot_store_path: str) -> None:
    results_dict = cluster_oknet_statistics(embedding_dict, blobs_folder_path=blobs_folder_path, model=model, num_clusters=10)
    embeddings = results_dict['kin_embeddings'].to_list()#.squeeze()
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings=embeddings.squeeze(1)
    
    print(embeddings.shape)
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP(n_neighbors=500, min_dist=0.3)
    #print(embeddings)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['skill']], alpha=0.2)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()

    le_gest = LabelEncoder()
    le_gest.fit(results_dict['gesture'])
    print('Generating gesture plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])], alpha=0.2)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()

    le_user = LabelEncoder()
    le_user.fit(results_dict['user'])
    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_user.transform(results_dict['user'])], alpha=0.2)
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_user.png')
    plt.savefig(save_path)
    # save xlims and ylims
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim() 
    print((xlim, ylim))
    plt.clf()
    
    # plot the embeddings for each user
    print('Generating user plots separately.')
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
    axs = axs.flatten()
    curr_ax = 0
    for user in results_dict['user'].unique():
        idxs = results_dict.query(f"user=='{user}'").index.to_list()
        axs[curr_ax].scatter(reduced_embeddings[idxs, 0], reduced_embeddings[idxs, 1], c=sns.color_palette()[curr_ax])
        #axs[curr_ax].set_aspect('equal', 'datalim')
        skill_dict = {"B": 0, "C": 1, "D": 2, "E": 2, "F": 1, "G": 0, "H": 0, "I": 0}
        axs[curr_ax].set_title(f'{user} (skill={skill_dict[user]})')
        curr_ax += 1
    save_path = os.path.join(plot_store_path, 'umap_user_separate.png')
    plt.setp(axs, xlim=xlim, ylim=ylim)
    fig.savefig(save_path)
    plt.clf()

def evaluate_model_multidata(
    blobs_folder_paths_list: str,
    model: OKNetV4,
    num_clusters: int,
    save_embeddings: bool,
    classifier_save_path: str = "./xgboost_save/multidata_xgboost.joblib",
) -> None:
    df = cluster_statistics_multidata(
        blobs_folder_paths_list=blobs_folder_paths_list, model=model, num_clusters=num_clusters
    )
    if save_embeddings:
        print("Saving dataframe.")
        df.to_pickle("./df.p")
    y = df["task"].values.ravel()
    X = [np.array(v) for v in df["opt_embeddings"]]
    X = np.array(X).reshape(-1, 2048)
    classifier = XGBClassifier(n_estimators=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5113)

    print("Fitting classifier.")
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)

    print("Training set classification report.")
    print(classification_report(y_train, y_hat))

    print("Test set classification report.")
    print(classification_report(y_test, y_hat_test))

    print("Saving classifier.")
    dump(classifier, classifier_save_path)
    print("Classifier saved.")


def evaluate_model_superuser(embedding_dict: dict, blobs_folder_path: str, model: OKNetV4, transcriptions_path: str, experimental_setup_path: str) -> None:
    transcription_file_names = os.listdir(transcriptions_path)
    transcription_file_names = list(filter(lambda x: '.DS_Store' not in x, transcription_file_names))
    
    transcription_translation_dict = {}
    count = 0
    transcription_file_names.sort()
    #print(transcription_file_names)
    #time.sleep(10000)
    for file in transcription_file_names:
        
        curr_file_path = os.path.join(transcriptions_path, file)
        with open(curr_file_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip()
                line = line.split(' ')
                start = line[0]
                end = line[1]
                gesture = line[2]
                
                transcription_name = file.split('.')[0] + '_' + start.zfill(6) + '_' + end.zfill(6) + '.txt'
                new_name = 'blob_{}_video'.format(count) + '_'.join(file.split('.')[0].split('_')[0:3]) + '_gesture_' + gesture +'.p'
                
                
                new_name = re.sub('Knot_Tying', '', new_name)
                new_name = re.sub('Needle_Passing', '', new_name)
                new_name = re.sub('Suturing', '', new_name)
                
                #print(f'converting {transcription_name} to {new_name}')
                #time.sleep(100000)
                
                transcription_translation_dict[transcription_name] = new_name
                
                #print(new_name)
                #print(transcription_name)
                
                count += 1
    #print(transcription_translation_dict)
    #time.sleep(1000000)
    df = cluster_oknet_statistics(embedding_dict, blobs_folder_path=blobs_folder_path, model=model, num_clusters=5)
    
    file_to_index_dict = {}
    file_count = 0
    for file in df['file_list']:
        file_to_index_dict[file] = file_count
        file_count += 1
    #print(file_to_index_dict)
    y = df['skill'].values.ravel()
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    
    X = [np.array(v) for v in df['kin_embeddings']]
    X = np.array(X).reshape(-1, 2048)
    
    sampler_list = []
    iterations = os.listdir(experimental_setup_path)
    iterations = list(filter(lambda x: '.DS_Store' not in x, iterations))
    iterations.sort(key=lambda x: x.split('_')[1])
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    itr = 0
    for iter_num in tqdm(iterations):
        directory_path = os.path.join(experimental_setup_path, iter_num)
        
        train_indices = []
        test_indices = []
        
        with open(os.path.join(directory_path, 'Train.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    #print(items[0])
                    #print(transcription_translation_dict[items[0]])
                    #pickle_file = transcription_translation_dict[items[0]]
                    #print(df[df['file_list']==pickle_file].index[0])
                    #print(file_to_index_dict[transcription_translation_dict[items[0]]])
                    
                    #time.sleep(10000)
                                        
                    train_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except Exception as e:
                    pass
                    #print(traceback.format_exc())
                    
            f.close()
        
        with open(os.path.join(directory_path, 'Test.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    test_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except:
                    pass
            f.close()
        #print(train_indices)
        #print(test_indices)
        
        #print(train_indices)
        #print('\n')
        #print(test_indices)
        #time.sleep(100000)
        
        X_train = X[train_indices]
        y_train = y_encoded[train_indices]
        X_test = X[test_indices]
        y_test = y_encoded[test_indices]

        classifier = XGBClassifier(n_estimators = 500)
        classifier.fit(X_train, y_train)

        # y_hat = classifier.predict(X_train)
        y_hat_test = classifier.predict(X_test)
        report_test = classification_report(y_test, y_hat_test, output_dict = True)
        # metrics['accuracy'] = (metrics['accuracy']*itr + report_test['accuracy'])/(itr + 1)
        # metrics['precision'] = (metrics['precision']*itr + report_test['weighted avg']['precision'])/(itr + 1)
        # metrics['recall'] = (metrics['recall']*itr + report_test['weighted avg']['recall'])/(itr + 1)
        # metrics['f1-score'] = (metrics['f1-score']*itr + report_test['weighted avg']['f1-score'])/(itr + 1)
        # metrics['support'] = (metrics['support']*itr + report_test['weighted avg']['support'])/(itr + 1)
        # itr += 1
        
        #y_hat_train = classifier.predict(X_train)
        #print(classification_report(y_train, y_hat_train))
        
        #print(classification_report(y_test, y_hat_test, output_dict = False))
        #print(confusion_matrix(y_test, y_hat_test))
        metrics['accuracy'].append(report_test['accuracy'])
        metrics['precision'].append(report_test['weighted avg']['precision'])
        metrics['recall'].append(report_test['weighted avg']['recall'])
        metrics['f1-score'].append(report_test['weighted avg']['f1-score'])
        metrics['support'].append(report_test['weighted avg']['support'])
        
        
    for key, val in metrics.items():
        print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))
        
    return np.mean(metrics['accuracy'])


def main():
    # Setup
    lr = 1e-3
    num_epochs = 1000
    weight_decay = 1e-8
    blobs_folder_path = Config.blobs_dir

    root = Config.trained_models_dir / "ok_network/HT5"
    if not root.exists():
        print(f"{root} does not contain a checkpoint")
        exit(0)

    # Load model
    net = OKNetV1(out_features=2048, reduce_kin_feat=False)
    # net = encoderDecoder(embedding_dim=2048)
    # net = net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss()
    checkpoint, net, optimizer = CheckpointHandler.load_checkpoint_with_model(
        root / "final_checkpoint.pt", net, optimizer
    )
    # net = net.opticalflow_net_stream
    embedding_path = root / "embedding_dict.pkl"
    if not embedding_path.exists():
        embedding_dict = store_embeddings_in_dict(blobs_folder_path=blobs_folder_path, model=net)
        pickle.dump(embedding_dict, open(embedding_path, "wb"))
    else:
        embedding_dict = pickle.load(open(embedding_path, "rb"))
    evaluate_model(
        embedding_dict, blobs_folder_path=blobs_folder_path, model=net, num_clusters=10, save_embeddings=False
    )
    # evaluate_model_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = 10, save_embeddings = False)


if __name__ == "__main__":
    main()