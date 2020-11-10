# Interpretable-COVID-Net

You can view the outputs in the output file repository accompanying
this repo. You can view example semantic dictionaires in 
  lucid_semantic_dictionary_distribution.ipynb and
  lucid_semantic_dictionary_magnitude.ipynb
 
 
to run CovidNet, GradCam or Lucid files in this repo
  install python3.6, virtualenv, and a CUDA version compatible with your GPU and tensorflow 1.15
  create a python virtual environment
    virtualenv env
  start your virtual environment
    windows: .\env\Scripts\activate
    linux: /env/bin/activate
  install the requirements
    pip install -r requirements.txt
  unzip the folders in the ouput repository

to run TCAV files in this repo
  clone the TCAV repository
  move the folder named 'covidnet' into the TCAV folder
    tcav-master/tcav/tcav_examples/image_models/covid
  create the folders
    tcav-master/projects/covidnet/activations
    tcav-master/projects/covidnet/cavs
    tcav-master/results
    
