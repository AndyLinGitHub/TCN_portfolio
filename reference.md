## 論文內容

- 緒論
  - 研究背景
  - 研究動機與目的
- 文獻回顧
  - 資產配置
    - Equal Weight
    - Markowitz
    - Risk Parity
  - 循環神經網路
    - RNN
    - LSTM
    - GRU
    - TCN
  - 損失函數
    - 可以預測價格、漲跌、報酬率排名、資產最佳權重
    - Mean squared error、Cross-entropy loss、...、Sharpe Ratio
- 研究方法
  - 模型架構
  - 損失函數
  - 訓練、驗證與測試框架
  - 模型效果評比
- 實證分析
  - 資產選擇
  - 實證結果
- 結論與建議
- 參考文獻

## 動機

- 資產配置、產業輪動的重要性
- 解決傳統資產配置方法的缺點
  - 模型參數的選擇對歷史資料最佳化

## 深度學習模型

- Time series問題適合以recurrent neural network架構 -> LSTM、GRU。而TCN在許多task上優於LSTM、GRU
- LSTM
  - LSTM
    - Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,
      9(8):1735–1780, 1997.
  - LSTM在投資、交易上的應用
    - Nelson, David & Pereira, Adriano & de Oliveira, Renato. (2017). Stock market's price movement prediction with LSTM neural networks.
    - Borovkova, S, Tsiamas, I. An ensemble of LSTM neural networks for high-frequency stock market classification. Journal of Forecasting. 2019
    - J. Sen, A. Dutta and S. Mehtab, "Stock Portfolio Optimization Using a Deep Learning LSTM Model," 2021 IEEE Mysore Sub Section International Conference (MysuruCon), Hassan, India, 2021, pp. 263-271
    - Yao, Haixiang and Li, Xiaoxin and Li, Lijun, Asset Allocation Based on Lstm and Black-Litterman Model.
- GRU
  - GRU
    - K. Cho, B. van Merrienboer, D. Bahdanau, and Y. Bengio. On the properties of neural machine
      translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259, 2014.
    - Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling, Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
  - GRU在投資、交易上的應用
    - Evaluation and Analysis of an LSTM and GRU Based Stock Investment Strategy, Zili Lin and Fangyuan Tian and Weiqian Zhang, Proceedings of the 2022 International Conference on Economics, Smart Finance and Contemporary
    - D. Lien Minh, A. Sadeghi-Niaraki, H. D. Huy, K. Min and H. Moon, "Deep Learning Approach for Short-Term Stock Trends Prediction Based on Two-Stream Gated Recurrent Unit Network," in IEEE Access, vol. 6, pp. 55392-55404, 2018
- TCN
  - TCN
    - Bai, Shaojie et al. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” 
  - TCN在投資、交易上的應用
    - Rui Zhang, Zuoquan Zhang∗, Marui Du and Xiaomin Wang. 2021. The Portfolio Model Based on Temporal Convolution Networks and the Empirical Research on Chinese Stock Market. In 2021 5th International Conference on Computer Science and Artificial Intelligence CSAI 2021), December 04-06, 2021, Beijing, China. ACM, New York, NY, USA, 10 Pages.
    - Wei Dai, Yuan An, Wen Long, Price change prediction of Ultra high frequency financial data based on temporal convolutional network, Procedia Computer Science, Volume 199, 2022, Pages 1177-1183, ISSN 1877-0509.

## 最佳化目標: 未來一段時間的Sharpe Ratio

- 使用Sharpe作為Loss Function訓練深度學習模型
  - Zhang, Zihao and Zohren, Stefan and Roberts, Stephen, Deep Learning for Portfolio Optimisation (May 29, 2020).
  - Enhancing Time-Series Momentum Strategies Using Deep Neural Networks, Bryan Lim, S. Zohren, Stephen J. Roberts, 2019, The Journal of Financial Data Science
- Sharpe Ratio 梯度
  - John Moody, Lizhong Wu, Yuansong Liao, and Matthew Saffell. Performance functions and reinforcement learning for trading systems and portfolios. Journal of Forecasting, 17(5-6):441– 470, 1998.
  - Gabriel Molina. Stock trading with recurrent reinforcement learning (RRL). CS229, nd Web, 15, 2016.
- Sharpe Ratio
  - W. F. Sharpe, “The sharpe ratio,” The Journal of Portfolio Management, vol. 21, no. 1, pp. 49–58, 1994.
- Optimizer
  - Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. Proceedings
    of the International Conference on Learning Representations, 2015.

## 方法比較對象

- Markowitz
- Risk Parity
- Equal Weight

## TODO

- 整理reference格式，要再搜一下每一個有沒有投上一些期刊，沒有就直接cite