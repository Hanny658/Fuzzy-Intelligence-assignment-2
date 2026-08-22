config
==========

-configure the parameter of  falcon dic

line 1: path to the training text file
line 2: path to the testing text file
line 3: test choice
	* 1 = iris
	* 2 = phoneme
	* 3 = traffic
	* 4 = spiral
	* 5 = cancer

line 4:

techni = not used , always = 1
option1 = choose termination of training
	* 1 = use error target
	* 2 = use MSE

option2 = choose method of calculating resonance
	* 1 = max input
	* 2 = avg input
	* 3 = fosart resonance model

option3 = choose learning model
	* 1 = fast learning
	* 2 = slow learning
	* 3 = soft to hard learning

testing configuration
=====================
1. iris
-----------
IrisTrain.txt
IrisTest.txt

2. phoneme
-----------
PhonemeTrain.txt
PhonemeTest.txt

3. traffic
-----------
TrafficTrain.txt
TrafficTest.txt

4. spiral
-----------
SpiralTrain.txt
SpiralTest.txt

5. Cancer
-----------
CancerTrain.txt
CancerTest.txt

6. Wisconsin diagnotic
----------------------
wdTrain.txt
wdTest.txt

testchoice = 5

7. Wisconsin Prognotic
----------------------
wp1Train.txt
wp1Test.txt

testchoice = 5

wp2Train.txt
wp2Train.txt 

testchoice = 3


trifdic
==========
-config the trifalcon-dic model

