<h2>TensorFlow-FlexUNet-Image-Segmentation-Acute-Lymphoblastic-Leukemia (2025/07/24)</h2>

This is the first experiment of Image Segmentation for Acute-Lymphoblastic-Leukemia based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1jqi4j9ntD9wgIhTxSRS9WK9WwudLCocA/view?usp=sharing">
Acute-Lymphoblastic-Leukemia-ImageMask-Dataset.zip</a>.
which was derived by us from <br>
<a href="https://www.kaggle.com/datasets/mehradaria/leukemia">Acute Lymphoblastic Leukemia (ALL) image dataset
</a>
<br>
<br>
<b>Acutual Image Segmentation for 512x512 Acute-Lymphoblastic-Leukemia images</b><br>
As shown below, the inferred masks look similar to the ground truth masks. 
The green region represents a benign, the cyan a malignant-early, and the yellow a malignant-pre respectively.<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Benign-009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Benign-009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Benign-009.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-285.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-285.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-285.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Pre-093.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Pre-093.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Pre-093.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>

The image dataset used here has been taken from the following kaggle web site.
<a href="https://www.kaggle.com/datasets/mehradaria/leukemia">Acute Lymphoblastic Leukemia (ALL) image dataset
</a><br>

If you use this dataset in your research, please credit the authors. <br>

<b>Data Citation:</b><br> 
Mehrad Aria, Mustafa Ghaderzadeh, Davood Bashash, Hassan Abolghasemi, Farkhondeh Asadi, and Azamossadat Hosseini,<br>
“Acute Lymphoblastic Leukemia (ALL) image dataset.” Kaggle, (2021).<br>
 DOI: 10.34740/KAGGLE/DSV/2175623.<br>
<br>
<b>Publication Citation:</b><br> 
Ghaderzadeh, M, Aria, M, Hosseini, A, Asadi, F, Bashash, D, Abolghasemi, H. <br>
A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification <br>
using peripheral blood smear images.<br>
 Int J Intell Syst. 2022; 37: 5113- 5133. doi:10.1002/int.22753<br>
<br>


<h3>
<a id="2">
2 Acute-Lymphoblastic-Leukemia ImageMask Dataset
</a>
</h3>
 If you would like to train this Acute-Lymphoblastic-Leukemia Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1jqi4j9ntD9wgIhTxSRS9WK9WwudLCocA/view?usp=sharing">
Acute-Lymphoblastic-Leukemia-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Acute-Lymphoblastic-Leukemia
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Acute-Lymphoblastic-Leukemia Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/Acute-Lymphoblastic-Leukemia_Statistics.png" width="512" height="auto"><br>
<br>
<!--
On the derivation of the dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
-->
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Acute-Lymphoblastic-Leukemia TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 5

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Acute-Lymphoblastic-Leukemia 1+4 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+4 classes
;categories = ["Benign", "Early", "Pre", "Pro"]
; RGB colors        Benign:green, Early:cyan,    Pre:yello,      Pro:red
rgb_map = {(0,0,0):0,(0,255,0):1,(0,255,255):2, (255, 255, 0):3, (255, 0,0):4}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 21,22,23)</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 44,45,46)</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 46 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/train_console_output_at_epoch46.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Acute-Lymphoblastic-Leukemia.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/evaluate_console_output_at_epoch46.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Acute-Lymphoblastic-Leukemia/test was not low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0285
dice_coef_multiclass,0.9835
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Acute-Lymphoblastic-Leukemia.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Benign-013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Benign-013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Benign-013.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Early-290.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Early-290.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Early-290.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Pre-106.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Pre-106.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Pre-106.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Pro-520.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Pro-520.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Pro-520.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Malignant-Pro-664.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Malignant-Pro-664.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Malignant-Pro-664.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/images/WBC-Benign-009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test/masks/WBC-Benign-009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Acute-Lymphoblastic-Leukemia/mini_test_output/WBC-Benign-009.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Acute Lymphoblastic Leukemia (ALL) image dataset. Kaggle, (2021)</b><br>
Mehrad Aria, Mustafa Ghaderzadeh, Davood Bashash, Hassan Abolghasemi, Farkhondeh Asadi, and Azamossadat Hosseini,<br>

 DOI: 10.34740/KAGGLE/DSV/2175623.<br>
<br>

<b>2. A fast and efficient CNN model for B-ALL diagnosis and its subtypes classification using peripheral blood smear images</b>
 <br>
Ghaderzadeh, M, Aria, M, Hosseini, A, Asadi, F, Bashash, D, Abolghasemi, H. <br>
<a href="https://onlinelibrary.wiley.com/doi/full/10.1002/int.22753">https://onlinelibrary.wiley.com/doi/full/10.1002/int.22753</a>
<br>
<br>
<b>3. Tensorflow-Image-Segmentation-Early-Acute-Lymphoblastic-Leukemia/b>
 <br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Early-Acute-Lymphoblastic-Leukemia-">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Early-Acute-Lymphoblastic-Leukemia-
</a>

