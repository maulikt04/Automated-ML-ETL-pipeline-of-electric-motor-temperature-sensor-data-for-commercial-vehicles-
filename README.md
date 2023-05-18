# Automated-ML-(ETL)-pipeline-of-electric-motor-temperature-sensor-data-for-commercial-vehicles (Master thesis)
--> Downstream of Data Analysis and Machine Learning 

### Research Objectives:
	The main research objective of this master’s thesis is to develop an automated machine learning pipeline for data wrangling (data gathering, data cleaning, and data structuring) and selecting the best model out of pre-defended multiple models during training for the production environment. 
	Use Based line linear regression, K-Neighbor regression, Ridge regression, Lasso regression, Decision Tree regression, and Random Forest to estimate the temperature of the rotor, Stator winding temperature, and mechanical torque. Derive the best features from the existing attributes by using feature engineering.
	To optimization of the created or existing machine learning pipeline.
	Evaluation of the model before and after adding new data which will create an effect on the model performance, applying the Validation methods on the pipeline, generating the graphical visualizations, and documentation of the result with brief explanations.



### Abstract:
Particularly in the manufacturing and electric car industries, permanent magnet synchronous motors (PMSM) are gaining popularity as the electric motor of choice for traction driving applications. To avoid issues linked to overheating, it is now necessary to monitor the temperatures of these crucial components. Recent research indicates that machine learning techniques are being applied more and more successfully in a variety of industries, including healthcare. Today, PMSM is being used in more and more applications, particularly in electric vehicle applications. The stator winding and rotor temperatures need to be closely monitored to minimize temperature spikes and make sure that the motor's performance is not compromised. 
In our thesis work, we will develop the machine learning pipeline because the manual machine learning process takes a lot of time compared to the automated machine learning process and requires manpower. In the industrial sector, data is constantly entering and being saved in the cloud. At that time, manual machine-learning processes were quite challenging to employ. The machine learning pipeline is crucial at that point to speed up the procedure. The goal of a pipeline is to scale up how many models you can realistically keep in production while allowing you to raise the iteration cycle with the increased confidence that codifying the process provides.
![image](https://user-images.githubusercontent.com/123557248/232788245-99bf3bdf-9c04-494d-81a3-af217f26bae8.png)


### Keywords:
Permanent magnet synchronous motors (PMSMs), Machine Learning Algorithms, Machine learning pipeline, Motor Temperature prediction, Graphical Visualizations

### Thesis work:
Our main aim in the master's thesis is to develop a machine-learning pipeline for analyzing sensor data. Despite the availability of various techniques, the process can be highly time intensive. The industrial sector involves non-stop manufacturing processes that produce a significant volume of data flowing through the pipeline. Nevertheless, it is recommended to keep this data in its raw and unlabeled form. Daily, we gather this data and employ preprocessing methods to evaluate it using different machine learning algorithms, which can be quite difficult to handle. Furthermore, there is a requirement for expanding the workforce to manage the current workload. In other words, our objective is to establish a collection of machine-learning models, which can be employed in a production environment, thereby enhancing operational efficiency and reducing the dependency on human resources.
In this context, we opted for a collection of temperature data about electric motors. Initially, we segregated this data into two separate Excel files consisting of present data and future data. Present data has already undergone full pre-processing and has been labeled, while the remaining future data has been unlabeled.
During the initial stage, we will construct an ML pipeline using present data. We will experiment with multiple models and train them using various splitting percentages such as 80% for training and 20% for testing. Afterward, we will assess the model's effectiveness on the present data. 
In the next stage, we intend to create a machine-learning pipeline utilizing the entire dataset. However, we are considering the unlabeled data as future data that we will encounter. Initially, we will label the unlabeled data and combine it with the previously labeled data using programming. Next, we will train and test the model on the newly created training and testing datasets, using the splitting percentages of 80% and 20%, respectively. Finally, we will assess the model's performance when faced with new data.
Our analysis will examine whether utilizing Present data for training multiple models will result in one model outperforming the others. For example, the first pipeline, which utilizes a Decision tree model trained on present data, is expected to yield favorable results. Later, we plan to include an additional future of previous data to train multiple models and wait for one model to produce desirable results. However, the decision tree algorithm doesn't need to provide good outcomes after using present data in the initial stage and again after utilizing 100% of the data. The model's performance might alter when new data is added in the future.
In the end, our primary objective is to create a machine-learning pipeline that is automated and can be applied to production data. The aim is to decrease the amount of time, workforce, and expenses involved in production.


### Motivation:
The rotor temperature, stator temperature, and torque are the most intriguing goal characteristics [1]-[6]. In a commercial vehicle, it is particularly difficult and expensive to correctly measure rotor temperature and torque. Strong rotor temperature estimators make it possible for the automotive industry to produce motors with less material and to implement control systems that make the most of the motor's potential. A precise estimation of the torque results in more accurate and appropriate motor control, which lowers power losses and eventually heat buildup.

### Programming Language:
Python

### Working Environment:
Google Colab and Jupyter Notebook

### References:
1. 	H. Guo, Q. Ding, Y. Song, H. Tang, L. Wang, and J. Zhao, “Predicting temperature of permanent magnet synchronous motor based on deep neural network,” Energies (Basel), vol. 13, no. 18, Sep. 2020, doi: 10.3390/EN13184782.
2. 	H. Zhang, M. Dou, and J. Deng, “Loss-minimization strategy of nonsinusoidal back EMF PMSM in multiple synchronous reference frames,” IEEE Trans Power Electron, vol. 35, no. 8, pp. 8335–8346, Aug. 2020, doi: 10.1109/TPEL.2019.2961689.
[3]	M. Jafari and S. A. Taher, “Thermal survey of core losses in permanent magnet micro-motor,” Energy, vol. 123, pp. 579–584, 2017, doi: 10.1016/j.energy.2017.02.016.
[4]	“5 Reasons Your Electric Motors Keep Overheating.” https://gesrepair.com/5-reasons-electric-motors-keep-overheating/ (accessed Apr. 03, 2023).
[5]	Ö. Çelik and S. S. Altunaydın, “A Research on Machine Learning Methods and Its Applications,” Journal of Educational Technology & Online Learning, vol. 1, no. 3, pp. 25–40, 2018, doi: 10.31681\jetol.457046.
[6]	“Advantages and Disadvantages of Machine Learning | Pros and Cons of Machine Learning, Drawbacks and Benefits - A Plus Topper.” https://www.aplustopper.com/advantages-and-disadvantages-of-machine-learning/ (accessed Apr. 03, 2023).
[7]	“Exploring the Advantages and Disadvantages of Machine Learning - TechVidvan.” https://techvidvan.com/tutorials/advantages-and-disadvantages-of-machine-learning/ (accessed Apr. 03, 2023).
[8]	G. D’aloisio and G. Stilo, “Modeling Quality and Machine Learning Pipelines through Extended Feature Models; Modeling Quality and Machine Learning Pipelines through Extended Feature Models,” 2022, doi: 10.1145/nnnnnnn.nnnnnnn.
[9]	“What is a Machine Learning Pipeline?” https://www.seldon.io/what-is-a-machine-learning-pipeline (accessed Apr. 03, 2023).
[10]	A. Lopez, A. Alberto, and L. Esquivel, “Electric Motor Data Analysis for Temperature Estimation,” 2021. [Online]. Available: https://www.researchgate.net/publication/352261170
[11]	T. V. Tran and E. Nègre, “Efficient estimator of rotor temperature designing for electric and hybrid powertrain platform,” Electronics (Switzerland), vol. 9, no. 7, pp. 1–12, Jul. 2020, doi: 10.3390/electronics9071096.
[12]	A. Zhou, C. Du, Z. Peng, Q. Peng, and D. Qin, “Rotor Temperature Safety Prediction Method of PMSM for Electric Vehicle on Real-Time Energy Equivalence,” Math Probl Eng, vol. 2020, 2020, doi: 10.1155/2020/3213052.
[13]	P. Thosar, J. Patil, M. Singh, S. Thamke, and S. Gonge, “Prediction of motor temperature using linear regression,” in Proceedings of the International Conference on Smart Technologies in Computing, Electrical and Electronics, ICSTCEE 2020, Institute of Electrical and Electronics Engineers Inc., Oct. 2020, pp. 7–12. doi: 10.1109/ICSTCEE49637.2020.9277184.
[14]	D. Huger and D. Gerling, “An advanced lifetime prediction method for permanent magnet synchronous machines,” Proceedings - 2014 International Conference on Electrical Machines, ICEM 2014, pp. 686–691, Nov. 2014, doi: 10.1109/ICELMACH.2014.6960255.
[15]	A. Gupta, “Prediction of Electric Motor Temperature (PMSM) Motor Using Decision Tree,” 2021. [Online]. Available: www.ijsrd.com
[16]	R. Savant, A. A. Kumar, and A. Ghatak, “Prediction and Analysis of Permanent Magnet Synchronous Motor parameters using Machine Learning Algorithms,” in Proceedings of 2020 3rd International Conference on Advances in Electronics, Computers and Communications, ICAECC 2020, Institute of Electrical and Electronics Engineers Inc., Dec. 2020. doi: 10.1109/ICAECC50550.2020.9339479.
[17]	T. Wu, H. Wang, and Y. Guo, “Thermal Modeling of Tubular Permanent Magnet Linear Synchronous Motor Based on Random Forest,” in 2021 13th International Symposium on Linear Drives for Industry Applications, LDIA 2021, Institute of Electrical and Electronics Engineers Inc., 2021. doi: 10.1109/LDIA49489.2021.9505794.
[18]	G. Kaya Uyanık and N. Güler, “A Study on Multiple Linear Regression Analysis,” Procedia Soc Behav Sci, vol. 106, pp. 234–240, 2013, doi: 10.1016/j.sbspro.2013.12.027.
[19]	D. V. Souza, J. C. Nievola, A. P. D. Corte, and C. R. Sanquetta, “k-Nearest Neighbor And Linear Regression In The Prediction Of The Artificial Form Factor,” Floresta, vol. 50, no. 3, pp. 1669–1678, 2020, doi: 10.5380/RF.V50I3.65720.
[20]	T. Hastie, R. Tibshirani, and J. Friedman, “Springer Series in Statistics The Elements of Statistical Learning Data Mining, Inference, and Prediction”.
[21]	“Ridge Regression - A Simple Tutorial for Beginners.” https://www.tutorialexample.com/ridge-regression-a-simple-tutorial-for-beginners/ (accessed Apr. 03, 2023).
[22]	A. Gelman and J. Hill, “Data Analysis Using Regression and Multilevel/Hierarchical Models,” Data Analysis Using Regression and Multilevel/Hierarchical Models, Dec. 2006, doi: 10.1017/CBO9780511790942.
[23]	R. Tibshirani, “Regression shrinkage and selection via the lasso: a retrospective,” J. R. Statist. Soc. B, vol. 73, pp. 273–282, 2011.
[24]	B. Sytnyk, “110399900”.
[25]	G. James, D. Witten, T. Hastie, and R. Tibshirani, “Springer Texts in Statistics An Introduction to Statistical Learning”, Accessed: Apr. 03, 2023. [Online]. Available: http://www.springer.com/series/417
[26]	“LASSO Regression: A Complete Understanding (2021) | UNext.” https://u-next.com/blogs/artificial-intelligence/lasso-regression/ (accessed Apr. 03, 2023).
[27]	S. K. Murthy, “Automatic Construction of Decision Trees from Data: A Multi-Disciplinary Survey”.
[28]	A. D. Gordon, L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, “Classification and Regression Trees,” Biometrics, vol. 40, no. 3, p. 874, Sep. 1984, doi: 10.2307/2530946.
[29]	“Chapter 11 Classification Algorithms and Regression Trees”.
[30]	L. Breiman, “Random forests,” Mach Learn, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.
[31]	A. Liaw and M. Wiener, “Classification and Regression by randomForest,” vol. 2, no. 3, 2002, Accessed: Apr. 03, 2023. [Online]. Available: http://www.stat.berkeley.edu/
[32]	“Electric Motor Temperature | Kaggle.” https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature (accessed Apr. 03, 2023).
[33]	“Correlation (Pearson, Kendall, Spearman) - Statistics Solutions.” https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/correlation-pearson-kendall-spearman/ (accessed Apr. 03, 2023).
[34]	B. G. Tabachnick and L. S. Fidell, “Using Multivariate Statistics Title: Using multivariate statistics,” 2019, Accessed: Apr. 03, 2023. [Online]. Available: https://lccn.loc.gov/2017040173
[35]	“The best Machine learning book: How to Learn Machine Learning.” https://howtolearnmachinelearning.com/books/machine-learning-books/the-book-to-start-you-on-machine-learning/ (accessed Apr. 03, 2023).
[36]	“Feature Selection For Machine Learning in Python - MachineLearningMastery.com.” https://machinelearningmastery.com/feature-selection-machine-learning-python/ (accessed Apr. 03, 2023).
[37]	I. Guyon and A. M. De, “An Introduction to Variable and Feature Selection André Elisseeff,” Journal of Machine Learning Research, vol. 3, pp. 1157–1182, 2003.
[38]	R. Kohavi and G. H. John, “Wrappers for feature subset selection,” Artif Intell, vol. 97, no. 1–2, pp. 273–324, Dec. 1997, doi: 10.1016/S0004-3702(97)00043-X.
[39]	S. S. Skiena, “The Data Science Design Manual”, doi: 10.1007/978-3-319-55444-0.
[40]	T. Hastie, R. Tibshirani, and J. Friedman, “Springer Series in Statistics The Elements of Statistical Learning Data Mining, Inference, and Prediction”.
[41]	“Introduction to machine learning by Ethem Alpaydin - PDF Drive.” https://www.pdfdrive.com/introduction-to-machine-learning-e166961950.html (accessed Apr. 03, 2023).
[42]	R. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” 1995, Accessed: Apr. 03, 2023. [Online]. Available: http://robotics.stanford.edu/~ronnyk
[43]	J. Bergstra, J. B. Ca, and Y. B. Ca, “Random Search for Hyper-Parameter Optimization Yoshua Bengio,” Journal of Machine Learning Research, vol. 13, pp. 281–305, 2012, Accessed: Apr. 04, 2023. [Online]. Available: http://scikit-learn.sourceforge.net.
[44]	“Google Vizier: A Service for Black-Box Optimization.” https://www.kdd.org/kdd2017/papers/view/google-vizier-a-service-for-black-box-optimization (accessed Apr. 04, 2023).
[45]	B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. De Freitas, “Taking the Human Out of the Loop: A Review of Bayesian Optimization”, Accessed: Apr. 04, 2023. [Online]. Available: http://www.ibm.com/software/commerce/optimization/cplex-optimizer/
[46]	H. Kita and Y. Sano, “Genetic Algorithms for Optimization of Noisy Fitness Functions and Adaptation to Changing Environments”.
[47]	M. Masum et al., “Bayesian Hyperparameter Optimization for Deep Neural Network-Based Network Intrusion Detection”.
[48]	A. Bilal, A. Waheed, and M. H. Shah, “A comparative study of machine learning algorithms for controlling torque of permanent magnet synchronous motors through a closed loop system,” 2019 2nd International Conference on Latest Trends in Electrical Engineering and Computing Technologies, INTELLECT 2019, Nov. 2019, doi: 10.1109/INTELLECT47034.2019.8955467.
[49]	K. Anuforo, “Temperature Estimation in Permanent Magnet Synchronous Motor (PMSM) Components using Machine Learning,” 2020.
[50]	“What is Machine Learning? | IBM.” https://www.ibm.com/topics/machine-learning (accessed Apr. 10, 2023).
 




