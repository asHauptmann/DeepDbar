%% Load and plot output of Dbar and Deep D-bar

load data/unitCirc mask maskMap

output=h5read(['data/DeepDbarOutput.h5'],'/result');
input=h5read(['data/DeepDbarOutput.h5'],'/imag');

for iii=1:size(output,3)
   
    DeepDbarOut=output(:,:,iii);
    DeepDbarIn =input(:,:,iii);

    DeepDbarOut(~mask)=nan;
    minVal=min(min(DeepDbarOut(mask)));
    maxVal=max(max(DeepDbarOut(mask)));

    figure(3)
    imagesc(DeepDbarIn)
    colorbar
    axis equal, axis off
    pause(1)

    figure(4)
    imagesc(DeepDbarOut)
    colorbar off
    axis equal, axis off
    set(gca,'clim',[minVal-0.02 maxVal])
    set(gcf,'Colormap', maskMap)
    
end






