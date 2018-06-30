%% DeepDbar script to run reconstruction of KIT4 data

% Accompanying code for the publication: Hamilton & Hauptmann (2018). 
% Deep D-bar: Real time Electrical Impedance Tomography Imaging with 
% Deep Neural Networks. IEEE Transactions on Medical Imaging.

% Sarah J. Hamilton and Andreas Hauptmann, June 2018
%% 1.) Run D-bar reconstruction

% Will follow soon
% Dbar_KIT4 
% imagesRecon=reconRect;
% save('data/DbarOutput.mat','-v7.3','imagesRecon')

%% 2.) Prepare data for CNN

%Specify network
filePath=['data/KIT4_CircTriSq_c12_19Jan.ckpt'];
%Input into network
dataSet  = ['data/DbarOutput.mat'];
%Output of network
fileOutName=['data/DeepDbarOutput.h5'];

%% 3.) Run CNN

% Command line to call python script and evaluation
systemCommand = ['python Eval_DeepDbar.py ' filePath ' ' fileOutName ' ' dataSet ]
[status, result] = system(systemCommand);

    
%% 4.) Display result

displayDeepDbarResult