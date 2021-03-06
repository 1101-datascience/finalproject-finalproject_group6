# [Group6] 信用卡流失客戶預測

### Groups
* 林書亞, 107302003
* 蔣明憲, 110753208
* 柯騰達, 110352011
* 張晏瑄, 106703056

### Goal
從信用卡留存/流失客戶的歷史資料中，挖掘影響用戶流失的重要因子，並預測目前用戶的流失機率

### Demo 
You should provide an example commend to reproduce your result
```R
Rscript Bank_Churn_Correct.R --input BankChurners.csv --modeloutput results/model_performance.csv --sampleoutput results/sample_performance.csv --fsoutput results/fs_performance.csv
```
* any on-line visualization

## Folder organization and its related information

### docs
* [Your presentation](docs/1101_datascience_FP_group6.pdf), by **Jan. 13**


### data

* Source：https://www.kaggle.com/sakshigoyal7/credit-card-customers
* Input format
  * 目標欄位：流失用戶標記（流失/留存）
  * 數值型欄位：年齡、家人數量、成為會員月數、目前擁有產品數、近一年不活躍月份數、近一年簽約數、信用額度、總循環信用、近一年月平均額度、總交易次數、總交易金額、第四季與第一季交易變化數、額度使用率
  * 類別型欄位：性別、教育程度、婚姻狀態、收入類別、信用卡等級
* Any preprocessing?
  * 欄位刪減：將相關性高的信用額度和近12個月平均額度擇其一留下（選擇保留12個月平均額度）
  * 數值調整：將總交易金額調整為平均交易金額
  * 收入類別資料中Unknown類別的處理：以$60-80K收入水準取代
  * 教育程度類別資料中Unknown類別的處理：以高中程度教育水準取代
  * 年齡變數（數值型）轉類別型：26 ~ 35、36 ~ 45、46 ~ 55、56 ~ 65、>65
  * 成為會員月數變數（數值型）轉類別型：0 ~ 10、11 ~ 20、21 ~ 30、31 ~ 40、>40
  * 樣本平衡：流失客戶樣本過少，採取SMOTE, MWMOTE方式產生流失客戶樣本

### code

* Which method do you use?
  * 模型：決策樹、隨機森林、XGBoost
  * 樣本產生方法：SMOTE, MWMOTE
* What is a null model for comparison?
  * 以資料的流失客戶比例作為其流失機率，計算pseudo R-Squared來對Null Model與最終模型進行比較
* How do your perform evaluation? ie. cross-validation, or addtional indepedent data set
  * 從原始資料中隨機抽樣20%資料，剩餘的80%資料透過5-fold cross-validation做驗證


### results

* Which metric do you use 
  * 三種模型比較指標：AUC 
  * 不同抽樣方法比較指標：Recall
* Is your improvement significant?
  * XGBoost的AUC高達0.9828，相較於決策樹的0.916，提升0.0668
  * XGBoost的AUC高達0.9828，相較於隨機森林的0.96，提升0.0228
  * SMOTE較未平衡資料，提升0.1096的Recall
  * pseudo R-Squared值高達0.7282，代表最終模型在預測上較Null Model精確許多
* What is the challenge part of your project?
  * 樣本極度不平均，雖然整體模型的precision很高，但預測到確實流失的客戶很少，再嘗試多種平衡樣本的方式後，才讓預測流失客戶的部分精準許多
  * 以Recall指標為準，進行樣本平衡的方法中，我們嘗試過Oversampling、Undersampling、ROSE、SMOTE以及MWMOTE。發現如果要極致的提高 recall 就是直接把多數類樣本做 Undersampling，然後少數的那部分做 Oversampling。所以純 Undersample 及 SMOTE 來說是最契合的，其次是 ROSE 跟 MWMOTE，兩者在提高不少Recall的情況下也能維持穩定Accuracy，Oversampling 則是小幅提高 Recall 而不影響 Accuracy，不進行處理則居於末位。
## References
* Code/implementation which you include/reference (__You should indicate in your presentation if you use code for others. Otherwise, cheating will result in 0 score for final project.__)
  * 決策樹：https://www.google.com/search?q=r+decision+tree&oq=r++de&aqs=edge.1.69i57j69i59j0i512l5j69i60j69i65.7924j0j4&sourceid=chrome&ie=UTF-8 
  * XGBoost:https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html
  * SMOTE: 1.https://iter01.com/480884.htmt 2.https://medium.com/%E6%95%B8%E5%AD%B8-%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E8%88%87%E8%9F%92%E8%9B%87/smote-enn-%E8%A7%A3%E6%B1%BA%E6%95%B8%E6%93%9A%E4%B8%8D%E5%B9%B3%E8%A1%A1%E5%BB%BA%E6%A8%A1%E7%9A%84%E6%8E%A1%E6%A8%A3%E6%96%B9%E6%B3%95-cdb6324b711e
  * MWSMOTE:https://www.twblogs.net/a/5b87b64d2b71775d1cd8a077
  * imbalance:https://rpubs.com/yoompubs/467234
* Packages you use
  *  rpart
  *  dplyr
  *  imbalance
  *  DMwR
  *  caret
  *  randomForest
  *  xgboost
  *  ROCR
  *  Ckmeans.1d.dp
  *  vip
  *  ggplot2
  *  pROC
