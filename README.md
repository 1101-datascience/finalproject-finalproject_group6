# [Group6] 信用卡流失客戶預測

### Groups
* 林書亞, 107302003
* name, student ID2
* name, student ID3
* ...

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
* Any related document for the final project
  * papers
  * software user guide

### data

* Source：https://www.kaggle.com/sakshigoyal7/credit-card-customers
* Input format
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
  * 
  * precision, recall, R-square
* Is your improvement significant?
* What is the challenge part of your project?

## References
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)
* Packages you use
* Related publications
