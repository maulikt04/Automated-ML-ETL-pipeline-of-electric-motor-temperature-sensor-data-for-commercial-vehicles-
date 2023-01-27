# Automated-ML-(ETL)-pipeline-of-electric-motor-temperature-sensor-data-for-commercial-vehicles (Master thesis)
--> Downstream of Data Analysis and Machine Learning

### Research Objectives:
1) To develop automated pipeline for data wrangling (data gathering, data cleaning and data 
structuring) and selecting the best model out of pre-defined multiple models during training for 
the production environment.
2) Optimization of the created or existed pipeline.
3) Evaluation and documentation of the results.


### Abstract:
Permanent magnet synchronous motors (PMSM) are becoming increasingly popular as the electric motor of choice for traction drive applications, particularly in the manufacturing and electric vehicle industries. This has created a need for monitoring the temperatures of these critical components to prevent overheating-related problems. According to recent studies, machine learning techniques are increasingly being used in various industries, such as healthcare, with positive outcomes. The use of PMSM as a motor is becoming more widespread today, especially in applications like electric vehicles. In order to prevent temperature increases and to ensure that the motor's performance does not suffer, the stator winding's temperature needs to be precisely monitored. In this work, we will develop the machine learning pipeline because of manual machine learning process takes a lot of time as compared to automated machine learning process and also need manpower in Manual machine learning process. In the industrial sectors, data is continuously coming and save in the cloud continuously. At that time very difficult to used manual machine learning processes. At that time machine learning pipeline is important for increase the speed of the process. Ultimately, the purpose of a pipeline is to allow you to increase the iteration cycle with the added confidence that codifying the process gives and to scale how many models you can realistically maintain in production.

### Keywords:
Permanent magnet synchronous motors (PMSMs), Machine Learning Algorithms, Machine learning pipeline, Motor Temperature prediction, Graphical Visualizations

### Thesis work:
In this work, we will develop machine learning pipeline in two different ways. First of all, we will develop machine learning pipeline with multiple models by using a 75% data from the dataset and train the multiple models and evaluating the different parameters according to the supervised learning case. 
We will consider another 25% data as an unlabeled data and we can assume as a new data which will come in the future. In second pipeline, we will create a small algorithm that works to convert unlabeled data to labeled data and add with old data. Again train multiple models and evaluating the different parameters according to the supervised learning case. 
We will analyzed, if we will use 75% data as a training of multiple models then any one model will give good result as compared to another models. For example, Decision tree gives good result in first pipeline which will train on 75% data.
Then after, we will add another 25% data with old data as a training of multiple models then one model again gives good result. But it is not compulsory decision tree gives good result by using 75% data on first phase then again decision tree give good result after using 100% data. The model may be change result after adding future upcoming data.
Ultimately, our main goal is to develop automated machine learning pipeline on production data and reduce the processing time, manpower and cost of production.

### Motivation:
The rotor temperature, stator temperature, and torque are the most intriguing goal characteristics [1]-[6]. In a commercial vehicle, it is particularly difficult and expensive to correctly measure rotor temperature and torque. Strong rotor temperature estimators make it possible for the automotive industry to produce motors with less material and to implement control systems that make the most of the motor's potential. A precise estimation of the torque results in more accurate and appropriate motor control, which lowers power losses and eventually heat buildup.

### Programming Language:
Python

### References:
1)  Ran Le, Kaijun He, Ace Hu “Motor Temperature Prediction with K-Nearest Neighbors and Convolutional Neural Network”.
2)  Andres Lopez, “Electric Motor Data Analysis for Temperature Estimation”. Conference Paper · June 2021, 1 st Andres Alberto L ´ opez Esquivel.
3) Hai Guo , Qun Ding 1,*, Yifan Song , Haoran Tang , Likun Wang and Jingying Zhao, “Predicting Temperature of Permanent Magnet Synchronous Motor Based on Deep Neural Network”. Journal of energies.
4) Aryan Gupta, Department of Electrical Engineering, Jamia Millia Islamia, New Delhi, India, “Prediction of Electric Motor Temperature (PMSM) Motor Using Decision Tree”.
5) Yuefeng Cen , Chenguang Zhang 1 , Gang Cen 1 , Yulai Zhang 1 and Cheng Zhao 2, “The Temperature Prediction of Permanent Magnet Synchronous Machines Based on Proximal Policy Optimization”.
6) Kenneth Anuforo, “Temperature Estimation in Permanent Magnet Synchronous Motor (PMSM) Components using Machine Learning.”
7) Germán H. Alférez, Oscar A. Esteban , Benjamin L. Clausen, Ana María Martínez Ardila, “Automated machine learning pipeline for geochemical analysis”.
8) Behrouz Derakhshan, Alireza Rezaei Mahdiraji, Tilmann Rabl, and Volker Markl, “Continuous Deployment of Machine Learning Pipelines”, Technische Universität Berlin.
9) Darius Roman , Saurabh Saxena, Valentin Robu , Michael Pecht and David Flynn, “Machine learning pipeline for battery state-of-health estimation”, Journal of nature machine intelligence.
10) Kaggle, Electric Motor Temperature estimation 185 hrs. Recordings from a permanent magnet synchronous motor (PMSM)



