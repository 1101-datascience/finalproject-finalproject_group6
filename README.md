# [Group6] 信用卡流失客戶預測

### Groups
* 林書亞, 107302003
* 蔣明憲, 110753208
* 柯騰達, 110352011
* 張晏瑄, 106703056

### Goal
從歷史信用卡留存/流失客戶的資料中，挖掘影響用戶流失的重要因子，並預測目前用戶的流失機率

### Demo 
You should provide an example commend to reproduce your result
```R
Rscript code/your_script.R --input data/training --output results/performance.tsv
```
* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1101_datascience_FP_<yourID|groupName>.ppt/pptx/pdf, by **Jan. 13**


### data

* Source：https://www.kaggle.com/sakshigoyal7/credit-card-customers
* Input format
  * 目標欄位：流失用戶標記(流失/留存)
  * 數值型欄位：年齡、家人數量、成為會員月數、目前擁有產品數、近一年不活躍月份數、近一年簽約數、信用額度、總循環信用、近一年月平均額度、總交易次數、總交易額、第四季與第一季交易變化數、額度使用率
  * 類別型欄位：性別、教育程度、婚姻狀態、收入類別、信用卡等級
* Any preprocessing?
  * 欄位刪減：將相關性高的信用額度和近12個月平均額度擇其一留下
  * 數值調整：將總交易金額調整為平均交易金額
  * 樣本平衡：流失客戶樣本過少，採取SMOTE, MWMOTE方式產生流失客戶樣本

### code

* Which method do you use?
  * 模型：決策樹、隨機森林、XGBoost
  * 樣本產生方法：SMOTE, MWMOTE
* What is a null model for comparison?
  * 以未做樣本平衡訓練出來的模型作為 null model
* How do your perform evaluation? ie. cross-validation, or addtional indepedent data set
  * 從原始資料中隨機抽樣20%資料，剩餘的80%資料透過5-fold cross-validation做驗證


### results

* Which metric do you use 
  * 三種模型比較：AUC 
  * 不同抽樣方法比較：Recall
* Is your improvement significant?
  * XGBoost的AUC高達xx，相較於決策樹的yy，提升zz
  * SMOTE較未平衡資料，提升xx Recall
* What is the challenge part of your project?
  * 樣本極度不平均，雖然整體模型的precision很高，但預測到確實流失的客戶很少，再嘗試多種平衡樣本的方式後，才讓預測流失客戶的部分精準許多

## References
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)
  * 決策樹：https://www.google.com/search?q=r+decision+tree&oq=r++de&aqs=edge.1.69i57j69i59j0i512l5j69i60j69i65.7924j0j4&sourceid=chrome&ie=UTF-8 
  * XGBoost:https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html
* Packages you use
  *  rpart
  *  dplyr
  *  imbalance
  *  DMwR
  *  ROSE
  *  caret
  *  randomForest
  *  ROCR
  *  xgboost
