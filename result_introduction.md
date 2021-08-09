## 
本文件对代码运行调试进行说明。

1、数据准备
  将比赛提供数据放置在指定路径下，本地路径：/data1/lvzhengwei/git_code/torch_dir/multi_modal/ccks/data/ccks2021
  运行  python extract_text_data.py，提取每个视频的文本数据。

  运行 python prepare_ccks_videotag.py  准备视频分类数据。  注意数据路径为本地路径（trainval_path，test_path，trainval_tsn_feature_dir)
  运行 python  prepare_ccks_semantic_tag.py  准备标签识别数据。  注意数据路径为本地路径（trainval_path，test_path)


2、分类模型训练、评估和预测。
  运行 python classify_tag_mul.py 文件，该文件包含训练、评估、预测三个函数，分别对应run_train,run_dev,run_predict函数。
  debug_flag 参数表示是否加载所有训练数据， debug_flag= True 表示加载小部分训练数据调试代码，debug_flag= False 表示加载全部训练数据。
  level 参数表示训练一级分类模型或者二级分类模型，  level = 'level1'表示一级分类模型，level = 'level2'表示二级分类模型
  data_version 参数表示数据切分的版本，用哪个版本的数据训练模型

3、语义标签识别模型训练、评估和预测。
  运行 python classify_sem.py文件。该文件包含训练、评估、预测三个函数，分别对应run_train,run_dev,run_predict_crf函数。
  debug_flag 参数表示是否加载所有训练数据， debug_flag= True 表示加载小部分训练数据调试代码，debug_flag= False 表示加载全部训练数据。
  data_version 参数表示数据切分的版本，用哪个版本的数据训练模型

## 多次重复1-3的过程，训练多个模型，进行集成。
4、模型集成和提交
  运行 python ccks_ensemble.py ，分布集成一级分类、二级分类、语义标签的结果。
  运行 python  generate_submission.py  生成最终结果，进行提交。


5、备注
  以上1-4过程，为比赛结果的复现过程。比赛各个单模型的结果、集成结果和最终提交结果见文件夹finial_result。
  finial_result/submision/result.txt为最终提交结果。

