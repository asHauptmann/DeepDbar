# DeepDbar loading module
#
# Accompanying code for the publication: Hamilton & Hauptmann (2018). 
# Deep D-bar: Real time Electrical Impedance Tomography Imaging with 
# Deep Neural Networks. IEEE Transactions on Medical Imaging.
#
# Sarah J. Hamilton and Andreas Hauptmann, June 2018


import numpy
import h5py # Read Matlab files

def extract_images(filename,imageName):
  """Extract the images into a 3D numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
  
  if inData.shape[0]==64:    
  
      rows = inData.shape[0]
      cols = inData.shape[1]
      print(rows, cols)
      data = numpy.array(inData)   
      data = data.reshape(rows, cols)  
      data=numpy.expand_dims(data,0)
      
  else:
      
      num_images = inData.shape[0]
      rows = inData.shape[1]
      cols = inData.shape[2]
      print(num_images, rows, cols)
      data = numpy.array(inData)
   
      data = data.reshape(num_images, rows, cols)
  
  return data


class DataSet(object):

  def __init__(self, images):
    """Construct a DataSet"""

    self._num_examples = images.shape[0]
    self._images = images

  @property
  def images(self):
    return self._images

def read_data_sets(FileName):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TEST_SET  = FileName
  IMAGE_NAME = 'imagesRecon'
    
  print('Start loading data')  
  
  test_images = extract_images(TEST_SET,IMAGE_NAME)
  data_sets.test = DataSet(test_images)

  return data_sets