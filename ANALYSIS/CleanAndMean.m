function [Mean_r,PositionWristsmoothed] = CleanAndMean(SpeedVelocity_Interp_r,PositionWristsmoothed)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(SpeedVelocity_Interp_r,2)
    I_ = find(~isnan(SpeedVelocity_Interp_r(:,i)) == 1);
SpeedVelocity_Interp_r(1:length(I_),i)= SpeedVelocity_Interp_r(I_,i);
SpeedVelocity_Interp_r(length(I_)+1:length(PositionWristsmoothed(:,:,1)),i) = NaN;
hold on
I_ =[];
end

Mean_r = nanmean(SpeedVelocity_Interp_r,2);
end

