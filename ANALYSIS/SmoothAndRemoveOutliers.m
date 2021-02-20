function [PositionWristsmoothed] = SmoothAndRemoveOutliers(PositionWristAligned,CoordinateOfInterest, Limit)
%Smoothing function and removal of "outliers" whic are artefacts due to the
%infrared signal of the camera, which affect both leap motion and the
%marker recording. But in Both case we Can remove it easily
%moving average coefficient found experimentaly 
for i = 1:size(PositionWristAligned,3)    
 for j = 1:3 % x, y, and z coordinate
PositionWristsmoothed(:,j,i) = smooth(PositionWristAligned(:,j,i),0.01,'moving');
 end
% First we removed the very obvious outliers
PositionWristsmoothed(find( PositionWristsmoothed(:,CoordinateOfInterest,i) < -1 * Limit),:,i) = NaN;

end

%then we removed the outloers that seems to appear at the very end of the
%recording. Here we loop through each data and if the data J+1 jump
%drastically in the z direction compared the data J, we say that this is an
%outliers

 for i = 1:size(PositionWristsmoothed,3)
for j = 2:length(PositionWristsmoothed)
if PositionWristsmoothed(j,3,i)< PositionWristsmoothed(j-1,3,i)/2
PositionWristsmoothed(j,3,i) = NaN;
end
end
end
end

