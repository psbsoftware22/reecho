# reecho
Now a days autoencoders is polpular in various field.  But, there is twoapplications  of  autoencoders  which  is  most  popular:   dimensionalityreduction and information retrieval.
* By using these features we can perform various image processing.* This project has four method of image processing
    * Generate Image
    * Noise Reduction
    * Constructing New Images
    * Colouring Images

#### Dataset:
* The dataset is from the below link:
http://vis-www.cs.umass.edu/lfw/lfw.tgz
#### Code File:
* All the code is used inside the reecho_model.ipynb notebook to run image processing.
* The autoencoder model is saved as variational_autoencoder.py.
#### Report:
* The report on this project is saved as report.pdf.
#### Run Test:
* To run test you need to run the test_model.py python file.
* Dowload the project and run the below command in the same directory
***
    python test_model.py
***
* Or, double click on the test_model.py file.

#### Results:
* Although the result is not promissing, the model works really well despite of lower amount of data and lower training time.
* By using some generated images and increase the image resolution we can achieve better result.
* AThe original and reconstructed data in 'image reconstruction' model
![image](https://drive.google.com/uc?export=view&id=1qxioB3VweZ_ic4RUoi2FNbAS6800EZ2y)
* The noise and reconstructed data in 'noise reduction' model
![image](https://drive.google.com/uc?export=view&id=12Rk-6XVnm8ZRX81NjDSrmEscDf_pyB0N)
* The newly constructed data in 'generating new images' model
![image](https://drive.google.com/uc?export=view&id=176ozTaJCjmX4Q30JNnV3Fz3PXrYd-1cs)
* The gray scale and reconstructed coloured images in 'colouring image' model
![image](https://drive.google.com/uc?export=view&id=1hqR80WNKr_QBAUFD4RXcpxFge4VsSbsm)