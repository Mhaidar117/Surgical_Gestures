# Multimodal learning

How to use this repository:
1) Extract optical flows from the video
2) Create data blobs
2) Train a model 
3) Evaluate the trained model and get different results including U-map plots, gesture classification, skill classification, task classification

This project does take a fair bit of disk space. I generated something close to 200 GB. It generates the optical flow entirely before hand to reduce the time to train. A few thoughts on how to save space: 
1) Delete the input to each step as it finishes. For example, once you generate the resized images, you no longer need the original. Once you generate the optical flow, you can delete the resized images, and once you generate the blobs, you can delete the optical flow. If you have the space, it might be worth keeping the optical flow folder around for the project.
2) The optical flow representation is the largest. If you can't store the whole dataset as optical flow at once, you can generate a subset of it as optical flow, generate the data blobs from it, and then delete the flow representation to save space
3) You can rewrite the code to calculate the optical flow after the blob generation. The current implementation is to generate flow for the whole video, and then extract the first 50 frames of it for the training. This improves flexibility at the cost of disk space. First extracting the frames and then calculating flow would also speed up processing time
4) If all else fails, use a subset of the data (sampling across skill levels). The comparison to the original paper will be weaker but having a homework to hand in is better than not. 

You can use a local machine to do steps 1 and 2 so you can inspect the data preprocessing steps and they are both CPU only jobs. The main steps are each a mode in main.py and we will be using main.py to call them. It is recommended to read through main.py and see what options there are and what default parameters you may be using to call each step. We are only setting the path parameters below, not changing the default scaling and other frame counts so use the default parameters in your analysis. If for some reason a run fails in the middle, you may need to delete the files that have been generated. It is not guaranteed that the code will overwrite the previously generate files or just append to them. 

### To install dependencies, please run

<code>pip install sklearn opencv-python tqdm pandas xgboost umap seaborn multipledispatch barbar</code>

### To generate the optical flow, run the following command. You may need to adjust the path for where you downloaded your JIGSAWS directory. This call will perform two steps. The first step will resize the video and the second will generate the optical flow. 

<code>python3 main.py --mode 'optical_flow' --source_directory ../JIGSAWS/Suturing/video/ --resized_video_directory ../JIGSAWS/Suturing/resized_video --destination_directory ../JIGSAWS/Suturing/optical_flow  </code> 

Make sure the code started running OK (it should say 'Rescaling videos; Processing video Suturing_B001_capture1.avi; 100; 200;...') then go get some coffee, take a nap. The "generating optical flow" step can take a while. In the mean time, once the resizing step is done, you can check that you have the same number of videos in the resized_video folder as the original video directory and you can play the videos (they should be smaller than the original though). For reference, on my 6-years-old desktop that had a pretty good CPU for when it came out, the resizing took about 3 min and the optical flow generation took about 1 h 20 min. 

### To generate prepare the data for training, you need to generate 'data_blobs'. This step pre-extracts the first 50 frames (subsampled by 2) for each gesture based on the transcriptions. This preprocessing step should speed up the training so that you don't need to do the kinematics extraction each time. This step took me about half an hour. It would've been faster to do this before the optical flow and only calculate flow for the relevant frames, but this way offers more flexibility in case you want to use a different number of frames.  

<code>python3 main.py --mode 'data_blobs' --optical_flow_path ../JIGSAWS/Suturing/optical_flow --transcriptions_path ../JIGSAWS/Suturing/transcriptions --kinematics_path ../JIGSAWS/Suturing/kinematics/AllGestures --blobs_path ../JIGSAWS/Suturing/blobs</code> 


### To train the network using your data_blobs. Run the following code. 

<code>python3 main.py --mode 'train' --blobs_folder_path '../JIGSAWS/Suturing/blobs' --weights_save_path models</code> 

The blobs folder path specifies where your data blogs are and the weights save path argument specifies the file path where you want the checkpoints of the models to be saved.

### You can train a different model by specifying the mode argument.

<code>python main.py --mode oknet_train --blobs_folder_path ../JIGSAWS/Suturing/blobs --weights_save_path ../models/oknet</code>

For example, to train the oknet, use the above code. 

### To evaluate the model, run the following code. 

<code>python main.py --mode oknet_eval --embedding_path ../embeddings/oknet_embeddings.p --blobs_path ../JIGSAWS/Suturing/blobs_whole --weights_save_path ../models/oknet_whole_2048/multimodal_Knot_Tying_epoch_400_2023-04-08_05_51_39.pth --model_dim 2048</code>

Again, the type of model evaluated can be changed in the mode argument. The --embedding path specifies where the embeddings pickle file is stored (if it already has been created for the given model). --weights_save_path specifies the pth file used to create the embeddings (if they are not created already)

### The following code evaluates all the saved models in the weights_save_folder 

<code>python main.py --mode multimodel_oknet_eval --embedding_path ../embeddings/oknet_embeddings.p --blobs_path ../JIGSAWS/Suturing/blobs_whole --weights_save_folder ../models/okwholecut --model_dim 2048</code>

### To evaluate one trial out and superuser (modes of cross validation), you can use the following codes

<code>python main.py --mode eval_superuser --embedding_path ../embeddings/oknet_embeddings.p --blobs_path ../JIGSAWS/Suturing/blobs_whole --weights_save_path ../models/oknet_whole_2048/multimodal_Knot_Tying_epoch_400_2023-04-08_05_51_39.pth --transcriptions_path ../JIGSAWS/Suturing/transcriptions --experimental_setup_path ../JIGSAWS/Experimental_setup/Suturing/Balanced/GestureClassification/OneTrialOut/1_Out</code>

<code>python main.py --mode eval_onetrialout_all --embedding_path ../embeddings/oknet_embeddings.p --blobs_path ../JIGSAWS/Suturing/blobs_whole --weights_save_path ../models/oknet_whole_2048/multimodal_Knot_Tying_epoch_400_2023-04-08_05_51_39.pth --transcriptions_path ../JIGSAWS/Suturing/transcriptions --experimental_setup_path ../JIGSAWS/Experimental_setup/Suturing/Balanced/GestureClassification/UserOut/</code>

In addition to the paths described previously, we provide the transcriptions path (so the script can remap the blobs to their frame indices) and the experimental setup path (so that the script can know what splits to use)

### Finally, to plot the maps, use the following code.

<code>python main.py --mode oknet_umaps --embedding_path ../embeddings/oknet_embeddings.p --blobs_path ../JIGSAWS/Suturing/blobs_whole --weights_save_path ../models/oknet_whole_2048/multimodal_Knot_Tying_epoch_700_2023-04-08_23_42_14.pth --plots_store_path ../plots/oknetv4/whole</code>


### Come up with your own command for evaluation. Look at the parameters that are required in main.py (lines 144-162) and see what needs to be filled in, and what parameters are used by default. 


The paper associated with this repository can be found at https://link.springer.com/article/10.1007/s11548-021-02343-y. The citation details are as follows.

@article{wu2021cross,
  title={Cross-modal self-supervised representation learning for gesture and skill recognition in robotic surgery},
  author={Wu, Jie Ying and Tamhane, Aniruddha and Kazanzides, Peter and Unberath, Mathias},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={16},
  number={5},
  pages={779--787},
  year={2021},
  publisher={Springer}
}

